# cluster_xchg_transport_ladder (all-exchange of one f32 per CTA within a cluster: four DSMEM transport spellings, their correctness constraints and SASS footprints)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS max_diff=0` (`xchg-push_mbar`, `xchg-push_bar`, `xchg-pull_all`, `xchg-pull_lane`), passed in two separate runs.** This entry gives no speed ranking between the transports; it records only the spellings, the constraints and what they lower to in SASS.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

When a cluster of CL CTAs (CL ≤ 8, the portable cap) must all-exchange "one f32 per CTA" into "every thread holds the cluster sum",
there are several ways to go between CTAs:

| Spelling | "Peer has launched" guarantee | How the data travels | Completion signal | Before exit |
|---|---|---|---|---|
| push_mbar | `barrier.cluster.arrive.relaxed` at the start, `barrier.cluster.wait` before sending | CL threads each issue 1 `st.async … mbarrier::complete_tx::bytes` to the peer's slot | local mbarrier (`expect_tx` = bytes this CTA will receive) | nothing |
| push_bar | same as above | CL threads each issue 1 `st.shared::cluster.f32` | a second `barrier.cluster.arrive.release` + `wait.acquire` | nothing |
| pull_all | merged with the completion signal into one barrier | write only its own slot; after the barrier each thread issues CL `ld.shared::cluster.f32` | that one release / acquire | `arrive` after reading, `wait` before exit |
| pull_lane | same as pull_all | each thread 1 `ld.shared::cluster` (lane & 7 picks the peer), merged by 3 levels of `__shfl_xor_sync` | same as pull_all | same as pull_all |

## SASS footprints (read instruction by instruction from the certification job's SASS)

| PTX | SASS (sequence read by the probe) |
|---|---|
| `barrier.cluster.arrive.relaxed.aligned` | a lone `UCGABAR_ARV` (wrapped in a `BRA.U` branch shell) |
| `barrier.cluster.arrive.release.aligned` | `MEMBAR.ALL.GPU` → `ERRBAR` → `CGAERRBAR` → `UCGABAR_ARV`; in another probe (each thread has just issued 8 `st.global` before the arrive) the sequence has an extra `MEMBAR.ALL.CTA` in front. Whether that is triggered by in-flight stores was not checked with a separate control |
| `barrier.cluster.arrive.aligned` (no qualifier) | identical instruction by instruction to `.release` — the default semantics is release |
| `barrier.cluster.wait.acquire.aligned` | `UCGABAR_WAIT` → `CCTL.IVALL` → `BRA` → `BAR.SYNC.DEFER_BLOCKING` (this `BAR.SYNC` is skipped by the unconditional `BRA` before it and is not on the execution path) |
| `mapa` + `st.async…complete_tx::bytes.b32` | `PRMT` assembles the remote address → `STAS` |
| `st.shared::cluster.f32` | generic `ST.E` (not `STS`) |
| `ld.shared::cluster.f32` | generic `LD.E` (not `LDS`; latency is that of a generic access); 8 in pull_all, 1 in pull_lane |
| `__shfl_xor_sync` | `SHFL.BFLY` |
| `mbarrier.try_wait.parity` loop | `SYNCS.PHASECHK.TRANS64.TRYWAIT` + branch |


## When needed (trigger)

- Reading the SASS / stall ledger of a cluster exchange segment and identifying, by the table above, which PTX each instruction came from.
- The exchange segment has a high share and you want to switch transport spelling: this entry gives only footprints and constraints; speed must be A/B'd in the same job on your own kernel.

## Minimal runnable example

```cpp
__device__ __forceinline__ uint32_t mapa_u32(uint32_t a, uint32_t r) {
  uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o; }
__device__ __forceinline__ float ld_cluster_f32(uint32_t a) {      // "memory" keeps it from being hoisted above the barrier
  float x; asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(x) : "r"(a) : "memory"); return x; }

// ---- pull_all (cluster 8)
__shared__ float slot0;
uint32_t pa[8];
for (int r = 0; r < 8; ++r) pa[r] = mapa_u32(smem_u32(&slot0), r);  // pure arithmetic, put it before any barrier
if (threadIdx.x == 0) slot0 = my_value;
asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");   // one barrier: peer launched + peer's slot written
asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
float s = 0.f;
for (int r = 0; r < 8; ++r) s += ld_cluster_f32(pa[r]);                  // the same summation order in every CTA
asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");   // my remote reads are done
/* ... use s ... */
asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");     // before exit: no peer is still reading me

// ---- push_mbar
__shared__ float slot[8]; __shared__ __align__(8) uint64_t bar;
if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  mbar_arrive_expect_tx(smem_u32(&bar), 8 * 4); }
__syncthreads();
asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // only announces "I have launched", no membar
/* ... compute my_value ... */
asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
if (threadIdx.x < 8)
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
               :: "r"(mapa_u32(smem_u32(&slot[rank]), threadIdx.x)), "r"(__float_as_uint(my_value)),
                  "r"(mapa_u32(smem_u32(&bar), threadIdx.x)) : "memory");
while (!mbar_try_wait(smem_u32(&bar), 0)) {}

```

## Gotchas

1. **Remote smem is accessible only while the peer has launched and not yet exited.** The push spellings rely on the early arrive at the start to guarantee "launched" and must
   `barrier.cluster.wait` before sending; in the pull spellings the data barrier also carries this guarantee, but one more barrier is required before exit (arrive after reading, wait before exit).
2. **The pull data barrier must be release / acquire** (the default semantics); a `.relaxed` arrive does not publish the store with which the peer wrote its own slot.
3. **A plain `st.shared::cluster` updates no mbarrier**; visibility relies only on the cluster barrier that follows.
4. **The asm of a remote read carries a `"memory"` clobber and is `volatile`**, otherwise the compiler may hoist the read above the barrier; `mapa` is pure arithmetic.
5. **A `.release` cluster arrive has a string of fence instructions more than `.relaxed`** (see the SASS table); an early arrive that only announces "launched" uses
   `.relaxed` (paired with `fence.mbarrier_init.release.cluster` to publish the barrier's init). **No qualifier means release**:
   `barrier.cluster.arrive.aligned` and `.release.aligned` lower to the same SASS string; to get relaxed you must write `.relaxed` explicitly.
   The fence cost grows with the writes in flight before the arrive: in one probe (each thread in the cluster has just issued 8 `st.global` before the arrive), the
   `clock64` cycles of the arrive itself (CTA median) were relaxed 41, no qualifier 856, `.release` 854; the whole launch 5.58 / 5.76 / 5.74 µs.
   In a whole kernel, changing the arrive of one push_mbar-style handshake (data travels by `st.async` + complete_tx, see the Purpose table) from no qualifier to `.relaxed`:
   the same-job lever pricing was 1.65% faster (three workloads +2.02 / +2.08 / +0.84%, noise 0.54 / 0.69 / 1.1% respectively, the third within noise;
   removing it alone from the full change set was 1.85% slower). `.relaxed` gives no ordering for any memory access before the arrive: when the peer, after its wait, reads data this thread
   wrote with ordinary stores before the arrive (pull spellings, or `st.shared::cluster` writes that the peer then reads), release is still required (gotcha 2). The probe's receive slots are dedicated smem; when a receive slot reuses smem that was just used for something else,
   relaxed does not guarantee the order between those pre-arrive accesses and the peer's writes; this is not probe-certified, verify it yourself.
6. **Every CTA in the cluster uses the same summation order / the same set of shuffles**, so that every CTA's sum is bitwise identical.
7. **The mbarrier-free spelling "value + epoch in one 8 B word, the receiver polls" did not pass on the CUDA probe and is not included in this entry.** The probe's `flag64` cell
   (`st.relaxed.cluster.shared::cluster.b64 {value, epoch}` pushed to the peer, the receiver polling its own slot with `ld.relaxed.cluster.shared::cta.v2.b64` until
   all 8 high words equal this launch's epoch) gave, in both jobs: the sum is right, but in every launch 144–288 threads never saw all
   epochs within the 2²² poll cap, cause not found; when two consecutive launches use the same epoch, the second is immediately satisfied by the stale words the first left behind (smem is not zeroed between launches). If you want an mbarrier-free
   handshake, write a probe and test it yourself first.

## Sources

- PTX ISA: `barrier.cluster.arrive{.release|.relaxed}.aligned`, `barrier.cluster.wait{.acquire}.aligned`, `mapa`,
  `ld.shared::cluster`, `st.shared::cluster`, `st.async`, `st.relaxed.cluster` / `ld.relaxed.cluster`,
  `mbarrier.arrive.expect_tx`, `fence.mbarrier_init`: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- CUDA C++ Programming Guide, Distributed Shared Memory (the peer must have launched before its smem is accessed remotely, and remote accesses must complete before exit;
  `cooperative_groups::cluster_group::map_shared_rank` is the public API for the same mechanism): <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUDA Binary Utilities (`cuobjdump -sass`): <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- GPU-certification probe — `PROBE PASS max_diff=0` (an earlier run passed as well): cluster 8, 64 CTAs x 128 threads, each CTA's value a small integer; under all four spellings every thread's sum equals the host's exactly; SASS sequences read per kernel (table above). `flag64` was `PROBE FAIL` both times (gotcha 7).
- GPU-certification probe (SASS and timing of the no-qualifier / `.release` / `.relaxed` arrive: the "no qualifier" row of the SASS table, second half of gotcha 5): cluster 4 x 128 threads, each thread's one b32 all-gathered via `st.async`,
  the three arrives numerically bit-identical — `PROBE PASS max_diff=0` (contracts build / data / sass-relaxed / sass-release all passed).
