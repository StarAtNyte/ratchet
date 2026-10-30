#ifdef DSA_WD
#include <cstdio>
#endif
// Minimal sm_100a PTX wrappers used by dsa.cu (mbarrier, cluster, TMA gather4, tcgen05, misc numerics).
#pragma once
#include <cstdint>

#define DI __device__ __forceinline__

namespace px {

DI uint32_t saddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ------------------------------------------------------------------ cluster
// cluster index along x of a pair launch, read fresh (keeps role code from reloading spilled prologue values)
DI int pair_index() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(r));
  return (int)(r >> 1);
}
DI int pair_count() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(r));
  return (int)(r >> 1);
}
DI uint32_t cta_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DI void cluster_arrive_wait() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// address of the same shared variable in CTA `rank` of the cluster
DI uint32_t peer_addr(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}

// ------------------------------------------------------------------ mbarrier
DI void bar_init(uint32_t b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
DI void bar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DI void bar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
// arrive on a barrier given by a shared::cluster address (local or peer)
DI void bar_arrive_cl(uint32_t b) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(b) : "memory");
}
DI void bar_arrive_cl_relaxed(uint32_t b) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(b) : "memory");
}
DI void bar_expect_tx_cl(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
#if defined(DSA_WD)
// watchdog (debug builds): report a wait that does not complete within ~2 s, then trap
DI void bar_wait(uint32_t b, uint32_t parity) {
  const long long t0 = clock64();
  while (true) {
    uint32_t r;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, 100000;\n\tselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(r) : "r"(b), "r"(parity) : "memory");
    if (r) return;
    const long long dt = clock64() - t0;
    if (dt > 2000000000ll && dt < 2000000000ll + 4000000ll && (threadIdx.x & 31) == 0)
      printf("WD block %d warp %d bar 0x%x parity %u\n", (int)blockIdx.x, (int)(threadIdx.x >> 5), b & 0xffff, parity);
    if (dt > 8000000000ll) __trap();
  }
}
#elif defined(DSA_SPIN)
DI void bar_wait(uint32_t b, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "W_%=:\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "\t@!p bra W_%=;\n}" ::"r"(b), "r"(parity)
      : "memory");
}
#else
DI void bar_wait(uint32_t b, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "W_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, 10000000;\n"
      "\t@!p bra W_%=;\n}" ::"r"(b), "r"(parity)
      : "memory");
}
#endif
// wait by polling with timed sleeps (no wake-ups on unrelated barrier traffic): for long waits off the critical path
DI void bar_wait_sleep(uint32_t b, uint32_t parity, uint32_t ns) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "W_%=:\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "\t@p bra D_%=;\n"
      "\tnanosleep.u32 %2;\n"
      "\tbra W_%=;\n"
      "D_%=:\n}" ::"r"(b), "r"(parity), "r"(ns)
      : "memory");
}
// wait with cluster-scope acquire (for barriers that receive arrivals from the peer CTA)
DI void bar_wait_cl(uint32_t b, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "W_%=:\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1, 10000000;\n"
      "\t@!p bra W_%=;\n}" ::"r"(b), "r"(parity)
      : "memory");
}

// non-blocking probe: has the phase with this parity completed?
DI bool bar_test(uint32_t b, uint32_t parity) {
  uint32_t r;
  asm volatile(
      "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n}"
      : "=r"(r) : "r"(b), "r"(parity) : "memory");
  return r != 0;
}
DI bool bar_test_cl(uint32_t b, uint32_t parity) {
  uint32_t r;
  asm volatile(
      "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n}"
      : "=r"(r) : "r"(b), "r"(parity) : "memory");
  return r != 0;
}

// ------------------------------------------------------------------ TMA
DI uint64_t pol_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
DI uint64_t pol_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// 4 rows x box-width columns of a 2-D tensor map -> 4 consecutive (swizzled) rows at dst; completion bytes go to
// `bar` (shared::cluster address, may be the peer CTA's barrier)
DI void gather4(uint32_t dst, const void* tmap, int col, int r0, int r1, int r2, int r3, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::"r"(dst),
      "l"(tmap), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(bar), "l"(pol)
      : "memory");
}
// 16-B global -> shared copy (L2 cache hint); completion tracked by cp_async_arrive
DI void cp_async16(uint32_t dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(pol) : "memory");
}
// arrive on a local barrier when all prior cp.async of this thread completed (no pending-count increment)
DI void cp_async_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(b) : "memory"); }
DI void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DI void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
DI void prefetch_tmap(const void* tmap) { asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory"); }
DI void l2_prefetch(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

// ------------------------------------------------------------------ global memory with cache hints
DI uint4 ld_stream(const void* p, uint64_t pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
DI void st_stream(void* p, uint4 v, uint64_t pol) {
  asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(p), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w), "l"(pol)
               : "memory");
}

// 32-B store (sm_100 256-bit vector access)
DI void st_stream32(void* p, const uint32_t* w, uint64_t pol) {
  asm volatile("st.global.L1::no_allocate.L2::cache_hint.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}, %9;" ::"l"(p),
               "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7]), "l"(pol)
               : "memory");
}

// ------------------------------------------------------------------ tcgen05
DI void tm_alloc2(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "r"(ncols));
}
DI void tm_relinquish2() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"); }
DI void tm_dealloc2(uint32_t t, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(t), "r"(ncols));
}
DI void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DI void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// D[tmem] (+)= A * B, pair of CTAs (issued by the even CTA)
DI void mma2_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n"
      "\ttcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
// same, A kept in the collector for the next MMA (fill) / taken from it and released (lastuse)
DI void mma2_ss_afill(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n"
      "\ttcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
DI void mma2_ss_alast(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n"
      "\ttcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
DI void mma2_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n"
      "\ttcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}" ::"r"(d),
      "r"(a), "l"(b), "r"(idesc), "r"(acc));
}
// arrive (once) on the barrier at offset `bar` in both CTAs of the pair when all prior MMAs completed
DI void mma2_commit(uint32_t bar) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar),
      "h"((uint16_t)3)
      : "memory");
}

// bf16 x bf16 -> fp32 instruction descriptor
constexpr uint32_t idesc_bf16(int M, int N, int a_mn, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) |
         ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
// shared-memory matrix descriptor, 128-byte swizzle
DI uint64_t sdesc_sw128(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) |
         (1ull << 46) | (2ull << 61);
}

// shared-memory matrix descriptor, no swizzle (8-row x 16-B core matrices; lbo: K direction, sbo: M/N direction)
DI uint64_t sdesc_none(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | (1ull << 46);
}

DI void tm_ld64(uint32_t a, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),
        "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]),
        "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]),
        "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
      : "r"(a));
}
DI void tm_ld32(uint32_t a, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(a));
}
DI void tm_st32(uint32_t a, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(a),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
      "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
      "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]));
}
DI void tm_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DI void tm_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// ------------------------------------------------------------------ misc
DI void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// named barrier that also ORs a predicate over the participating threads
DI bool nbar_or(int id, int n, bool v) {
  uint32_t r;
  asm volatile("{\n\t.reg .pred p, q;\n\tsetp.ne.u32 q, %3, 0;\n\tbarrier.cta.red.or.pred p, %1, %2, q;\n\tselp.u32 %0, 1, 0, p;\n}"
               : "=r"(r) : "r"(id), "r"(n), "r"((uint32_t)v) : "memory");
  return r != 0;
}
DI void nbar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DI void nbar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
template <int N> DI void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N)); }
template <int N> DI void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N)); }
DI bool elect() {
  uint32_t p;
  asm volatile("{\n\t.reg .pred P;\n\telect.sync _|P, 0xffffffff;\n\tselp.b32 %0, 1, 0, P;\n}" : "=r"(p));
  return p != 0;
}

DI float exp2a(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DI float log2a(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// paired fp32 ops
DI uint64_t pk2(float a, float b) {
  uint64_t r;
  asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b));
  return r;
}
DI void up2(uint64_t v, float& a, float& b) { asm("mov.b64 {%0, %1}, %2;" : "=f"(a), "=f"(b) : "l"(v)); }
DI uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
}
DI uint64_t add2(uint64_t a, uint64_t b) {
  uint64_t r;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}
DI uint64_t mul2(uint64_t a, uint64_t b) {
  uint64_t r;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}
// (lo, hi) -> bf16x2 with lo in the low half
DI uint32_t bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
// (a0, a1, a2, a3) -> 4 e4m3 bytes, a0 in the low byte
DI uint32_t e4m3x4(float a0, float a1, float a2, float a3) {
  uint32_t r;
  asm("{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n\tcvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n\t"
      "mov.b32 %0, {lo, hi};\n\t}" : "=r"(r) : "f"(a0), "f"(a1), "f"(a2), "f"(a3));
  return r;
}
// (lo, hi) -> e4m3x2 with lo in the low byte
DI uint32_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

}  // namespace px
