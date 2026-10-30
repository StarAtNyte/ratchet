# Hopper CUDA knowledge for this local campaign

Target: H200, compute capability 9.0a. Confirm all timing and occupancy claims on the actual target. The Muse challenge uses fresh inputs, CUDA graph replay and cold L2; a local campaign is a screen and its candidates require the challenge's official correctness and timing checks. Preserve BF16 activations, FP32 residual streams, and exact permitted weight precision.

The Muse prefill boundary implementation reads an 8192-element BF16 branch and FP32 stream per row, performs two FP32 RMS reductions and a residual mix, and emits BF16 normalized output plus FP32 stream. Earlier H200 component screens found a 256-thread streaming-hint configuration about 0.9–1.4% faster for one recompute boundary; full-layer gains were not established. Measure alternatives on all campaign shapes and do not infer a full-layer gain from component timing.
