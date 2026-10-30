# pdl_griddepcontrol_launch (Programmatic Dependent Launch: two consecutive kernels on the same stream hand off with `griddepcontrol`, the second one's launch / prologue overlapping the first one's tail)

> **Certification source: CUDA-mode B200 GPU probe (not included in this release), all three contracts PASS
> (`pdl-numeric` `PROBE PASS max_diff=0`, `pdl-early-launch` `PROBE PASS max_diff=n/a`, `pdl-sass` `PROBE PASS max_diff=n/a`).**
> Certification covers: numerics (including data written only after the trigger), the dependent's prologue preceding the end of the first kernel under an early trigger, and the SASS footprint; the chained timing in the probe is only a fact line.
> Whole kernel: in the same class of two-kernel flow, adding PDL to the second kernel made all three workloads about 5.5–6.9% faster (see "When needed").
>
> One file per hardware interface. Operator-agnostic: mechanism only. CUDA mode: CUDA runtime `cudaLaunchKernelEx` + inline PTX `griddepcontrol`.
> For how to issue a batch of loads in one go after the wait see [`post_wait_gather_single_trip`](post_wait_gather_single_trip.md).

## Purpose

When kernel B on the same stream reads kernel A's output, a plain `<<<>>>` launch makes B wait until A's **whole grid has exited and memory is flushed** before B starts being scheduled;
in between is a launch gap plus B's prologue time. PDL lets B start being scheduled once every CTA of A has executed `griddepcontrol.launch_dependents` (or exited);
B first runs the prologue that does not depend on A, and at `griddepcontrol.wait` it waits for A to complete with memory visible. That way, the launch gap of the short later kernel in a multi-kernel flow
hides in the tail of the earlier kernel (putting prologue work before the wait has no certified gain, see gotcha 1).

## When needed (trigger)

- One call launches two or more kernels back to back, the later one reads global data the earlier one wrote, and the later one is launched with `<<<>>>`; there is no `griddepcontrol` in the source.
- nsys / NCU show a gap of a few µs between the two kernels, while the later kernel itself takes only a few to a few dozen µs.
- Accept by SASS: A has the trigger instruction, B has the wait instruction (gotcha 7); whether the host really passes the attribute must be confirmed by timing / inter-kernel spacing.
- **Whole-kernel readings** (the same class of two-kernel flow: a main kernel followed by a short kernel reading its output; single-lever same-job pricing, three workloads): adding PDL only to the second kernel
  (trigger at the end + wait before the first read + attribute launch) was +6.05% faster (per workload +5.71% / +6.93% / +5.51%, noise 0.5–1.3%), and removing it from the full lever set was −5.30% slower;
  after landing, the gate device geometric mean went 0.051 → 0.0483 ms (confirmation job 0.0483). On the same kernel, moving the trigger earlier from the end made the whole kernel slower at every position tried
  (−1.74% to −3.4%, see gotcha 3); moving a piece of B's setup before the wait was also slower (−1.26%, see gotcha 1).

## Minimal runnable example

```cpp
// Producer A: a plain launch is fine (it needs no attribute itself)
__global__ void kA(int* buf /*...*/) {
  // ... body: compute and write buf (global) ...
  // place it after this CTA's last write of global data that B reads (the most conservative position, see gotchas 2, 3);
  // B is scheduled only once at least one thread of every CTA has reached it (or the CTA has exited)
  asm volatile("griddepcontrol.launch_dependents;");
}

// Consumer B: wait once before the first load that reads A's output
__global__ void kB(const int* buf, int* out /*...*/) {
  // prologue: put only work that does not read A's output before the wait; whether and how much to move must be timed yourself (gotcha 1)
  asm volatile("griddepcontrol.wait;" ::: "memory");   // "memory" clobber: keeps the compiler from hoisting loads of buf above this
  out[threadIdx.x] = __ldcg(buf + threadIdx.x);        // A's output is read only from here on; the probe uses __ldcg, plain loads untested (gotcha 8)
}

// host: A with <<<>>>, B with cudaLaunchKernelEx + the attribute, both on the same stream
void launch(int* buf, int* out, int gridA, int gridB, cudaStream_t st) {
  kA<<<gridA, 128, 0, st>>>(buf);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(gridB);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = 0;            // B's dynamic smem goes here, not in the third <<<>>> argument
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;                    // set to 0 it degrades to a plain stream-ordered launch, handy for an A/B control
  cudaError_t e = cudaLaunchKernelEx(&cfg, kB, (const int*)buf, out);   // argument types must match kB's parameters exactly (template deduction)
  // check e (gotcha 6)
}
```

In a PyTorch extension, `st` is `at::cuda::getCurrentCUDAStream()`, and both A and B are launched on that stream.

## Gotchas

1. **Missing `griddepcontrol.wait` in B is a data race**: once B is launched with the attribute it may start reading while A is still writing. The probe's negative control
   (early trigger + attribute + no wait) read stale values for 18944 / 18944 elements all three times. Put the wait **only once** per kernel, before the first load that reads A's output.
   Work that does not depend on A may go before the wait, but that has no certified gain: the probe's B prologue is a single timestamp; in the same class of two-kernel flow, changing a piece of B's setup
   to be recomputed from the inputs before the wait was 1.26% slower on the whole kernel as a single lever (three workloads −1.16% / −1.31% / −1.30%, noise 0.5–0.6%), and 1.08% slower combined with an earlier trigger.
   Put the wait at B's entry by default; time it yourself before moving prologue work ahead of it.
2. **The trigger position only decides when B can be scheduled, not visibility**: `griddepcontrol.wait` waits for A's whole grid to **complete with memory flushed**, not for the trigger.
   The probe's `pdl-numeric` deliberately has A write data only **after** the trigger; 5 runs each of early and end-of-kernel trigger, and B read everything correctly after the wait (max_diff=0).
3. **An early trigger competes with A for resources**: once B's CTAs are scheduled they occupy registers / smem / warp slots and idle on the wait. In the probe (A and B both small enough to co-reside)
   with an early trigger B's prologue was about 196 µs earlier than A's end, and the chained timing of early / end-of-kernel trigger was flat (5.51 / 5.54 µs per pair); but on a whole kernel (the same class of two-kernel flow),
   moving the trigger earlier from A's end was slower at every position: to the entry, −2.57% as a single lever (three workloads −2.32% / −2.91% / −2.49%), two whole-kernel landings (cross-job)
   −3.2% / −3.4%; to the start of A's last stretch of work, −1.74% as a single lever (−1.53% / −1.92% / −1.76%), three stretches earlier still −2.52% (−2.48% / −2.65% / −2.42%).
   Put it after A's last global write by default; time it yourself before moving it earlier.
4. **A without a trigger is also correct** (Programming Guide: the implicit trigger then happens when all of A's CTAs exit, missing the overlap with A's tail; the probe did not test this setting).
   Conversely, **when B carries no attribute**, `<<<>>>` / `numAttrs = 0` is plain stream order: in the probe's plain-launch control B's prologue always came after A's end (+1.9 to 3.9 µs),
   so writing the instructions in the kernel while the host still uses `<<<>>>` amounts to not doing PDL.
5. **The attribute goes on B's (the dependent's) launch, not A's.** A and B must be launched consecutively on the same stream (Programming Guide; the case with other operations in between was not tested by the probe).
   For a three-stage chain A→B→C, B can itself execute `launch_dependents` and C is launched with the attribute (Programming Guide usage, not tested by the probe).
6. **`cudaLaunchKernelEx` is a template**: arguments must match the kernel parameters exactly, one by one (passing `int*` to a `const int*` parameter needs an explicit cast), otherwise compilation fails with a template deduction error.
   Check the return value (the probe checked it with `TORCH_CHECK` only on the success path; the behavior on devices that do not support the attribute was not tested).
7. **Accept by SASS, not by source**: in `cuobjdump -sass`, A's `griddepcontrol.launch_dependents` and B's `griddepcontrol.wait`
   each lower to one dedicated instruction: measured `PREEXIT` on A's side and `ACQBULK` on B's (if these two mnemonics are not in the SASS, it did not take effect). Whether the host really passes the attribute for B
   cannot be seen in SASS; confirm with timing / nsys inter-kernel spacing (probe `pdl-early-launch`: with the attribute + early trigger, B's prologue timestamp is earlier than A's end timestamp,
   which never happens with a plain launch).
8. **The `"memory"` clobber must not be omitted** (on the wait). Without it, the compiler may schedule ordinary loads of A's output before the wait (compiler semantics; the probe did not run a negative control without the clobber).
   The probe reads A's output with `__ldcg`; plain loads were not tested (the PTX ISA semantics of wait is that the prerequisite grid's memory operations are complete and visible to the current grid, with no `.cg` requirement).
9. The runtime API has the equivalent `cudaTriggerProgrammaticLaunchCompletion()` / `cudaGridDependencySynchronize()` (sm_90+),
   the same mechanism as the two PTX instructions; neither instruction may appear on a `__CUDA_ARCH__ < 900` compile path.

## Readings (probe; B200, SM clock locked at 1500 MHz)

- B prologue timestamp − A end timestamp (5 runs): early trigger + attribute −195.9 to −197.1 µs (A spins about 200 µs after the trigger); plain launch +1.9 to 3.9 µs;
  end-of-kernel trigger + attribute +0.5 to 0.7 µs.
- 200 back-to-back A→B pairs (A not spinning): plain 7.99 µs / pair, attribute + end-of-kernel trigger 5.54 µs / pair, attribute + early trigger 5.51 µs / pair.
- Negative control (early trigger + attribute, B without wait): 18944 / 18944 elements read stale values (identical all three times).
- SASS: A's trigger → `PREEXIT` (one each at the early / end positions), B's wait → `ACQBULK`.

## Sources

- PTX ISA, `griddepcontrol` (`.launch_dependents` / `.wait`): <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol>
- CUDA C++ Programming Guide, Programmatic Dependent Launch and Synchronization (`cudaLaunchKernelEx`,
  `cudaLaunchAttributeProgrammaticStreamSerialization`, `programmaticStreamSerializationAllowed`,
  `cudaGridDependencySynchronize`, `cudaTriggerProgrammaticLaunchCompletion`):
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization>
- GPU-certification probe (contracts `pdl-numeric` / `pdl-early-launch` / `pdl-sass`):
  `PROBE PASS max_diff=0`, `PROBE PASS max_diff=n/a`, `PROBE PASS max_diff=n/a`.
