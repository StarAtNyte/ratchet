<div align="center">

# ForgeKernel 算子代码

**ForgeKernel 在 NVIDIA H100 与 B200 上生成的 GPU 算子代码**

[English](README.md) | **中文**

</div>

本目录公开论文
[*ForgeKernel: Self-Evolving Kernels Catching and Surpassing Experts and Agents*](https://openreview.net/pdf?id=EtV3f5BPD4)
（ICLR 2027 审稿中）里由 ForgeKernel 生成的算子（kernel）代码，以及论文投稿之后 ForgeKernel 新生成的一个算子，
后者在下表中标为「论文之后」。

ForgeKernel 是一个自进化的 kernel agent。它只通过硬件行为读出参照 kernel 的架构：对一个封装好的
二进制做运行、计时和 profiling，从不读它的源码。读出的架构被整理成一份架构优先的课程，交给 coding
agent 实现。每个被接受的 kernel 都可以封装起来，晋升为下一轮的参照，因此优化可以越过已知最好的实现继续推进。
论文里的 kernel 都用 [CuTe DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html)
（`nvidia-cutlass-dsl`）编写，论文之后的那个算子是 CUDA C++。所有 kernel 在生成时都看不到对比对象（库或其他系统）的源码。

## 目录结构

先按算子类型分类，再按算子和 GPU 分目录：

```
kernels/
├── attention/          稠密 attention：前向、反向、paged decode/prefill
├── sparse_attention/   DeepSeek 稀疏 attention（DSA）与 DeepSeek-V4.1 融合 prefill
├── linear_attention/   Gated DeltaNet 与 Kimi Delta Attention
├── gemm/               稠密、FP8 与 FP8 × FP4 grouped GEMM
└── conv/               二维与三维卷积
```

每个叶子目录放一个 kernel（`kernel.py`；论文之后的算子另附 CUDA 源文件）。

## 算子

数字为正表示 ForgeKernel 的 kernel 更快。「出处」指数字来自哪里：论文 Figure 3（对比专家库）、
Figure 4（对比其他 kernel 生成系统）、Table 1（参照晋升带来的自进化），或论文之后。

### Attention

| 算子 | GPU | 对比对象 | 加速 | 出处 | 目录 |
|-|-|-|-|-|-|
| Attention 前向，causal，GQA 16:1，第 0 代 | H100 | FlashAttention-4 | +0.3% | Figure 3 | [`attention/attn_fwd_gqa16/h100/gen0`](attention/attn_fwd_gqa16/h100/gen0/) |
| Attention 前向，causal，GQA 16:1，第 1 代 | H100 | ForgeKernel 第 0 代（封装） | +9.9%（比 FlashAttention-4 快 10%） | Table 1 | [`attention/attn_fwd_gqa16/h100/gen1`](attention/attn_fwd_gqa16/h100/gen1/) |
| Attention 前向，causal，GQA 16:1，第 2 代 | H100 | ForgeKernel 第 1 代（封装） | +4.2%（比 FlashAttention-4 快 15%） | Table 1 | [`attention/attn_fwd_gqa16/h100/gen2`](attention/attn_fwd_gqa16/h100/gen2/) |
| Attention 前向，causal，GQA 16:1 | H100 | /goal agent 的 kernel（封装） | +5.0%（比 FlashAttention-4 快 29%） | Table 1 | [`attention/attn_fwd_gqa16/h100/from_agent`](attention/attn_fwd_gqa16/h100/from_agent/) |
| Attention 反向，causal，MHA | H100 | FlashAttention-4 | +5.1% | Figure 3 | [`attention/attn_bwd_mha/h100`](attention/attn_bwd_mha/h100/) |
| Attention 反向，causal，MHA | B200 | FlashAttention-4 | +2.5% | Figure 3 | [`attention/attn_bwd_mha/b200`](attention/attn_bwd_mha/b200/) |
| Attention 反向，causal，GQA 8:1 | H100 | FlashAttention-4 | +2.7% | Figure 3 | [`attention/attn_bwd_gqa8/h100`](attention/attn_bwd_gqa8/h100/) |
| GQA paged decode | H100 | K-Search | +24.5% | Figure 4 | [`attention/gqa_paged_decode/h100`](attention/gqa_paged_decode/h100/) |
| MLA paged decode | H100 | K-Search | +170.6% | Figure 4 | [`attention/mla_paged_decode/h100`](attention/mla_paged_decode/h100/) |
| MLA paged prefill | H100 | K-Search | +146.1% | Figure 4 | [`attention/mla_paged_prefill/h100`](attention/mla_paged_prefill/h100/) |

### 稀疏 attention

| 算子 | GPU | 对比对象 | 加速 | 出处 | 目录 |
|-|-|-|-|-|-|
| DSA 稀疏 attention 反向 | H100 | cuDNN-frontend | +1.4% | Figure 3 | [`sparse_attention/dsa_sparse_bwd/h100`](sparse_attention/dsa_sparse_bwd/h100/) |
| DSA score 重算 | H100 | cuDNN-frontend | +6.2% | Figure 3 | [`sparse_attention/dsa_score_recompute/h100`](sparse_attention/dsa_score_recompute/h100/) |
| DSA indexer 反向 | H100 | cuDNN-frontend | +2.5% | Figure 3 | [`sparse_attention/dsa_indexer_bwd/h100`](sparse_attention/dsa_indexer_bwd/h100/) |
| DeepSeek-V4.1 融合 prefill：Q RoPE + 带 sink 的稀疏 MQA attention + O RoPE + FP8 cast | B200 | FlashMLA | 1.104–1.105× | 论文之后 | [`sparse_attention/dsv41_fused_attn_rope_cast/b200`](sparse_attention/dsv41_fused_attn_rope_cast/b200/) |

### 线性 attention

| 算子 | GPU | 对比对象 | 加速 | 出处 | 目录 |
|-|-|-|-|-|-|
| Gated DeltaNet prefill 前向，第 0 代 | H100 | FlashInfer | −0.2% | Figure 3 | [`linear_attention/gdn_prefill_fwd/h100/gen0`](linear_attention/gdn_prefill_fwd/h100/gen0/) |
| Gated DeltaNet prefill 前向 | H100 | /goal agent 的 kernel（封装） | +9.5%（比 FlashInfer 快 100%） | Table 1 | [`linear_attention/gdn_prefill_fwd/h100/from_agent`](linear_attention/gdn_prefill_fwd/h100/from_agent/) |
| Gated DeltaNet prefill 反向 | H100 | FlashQLA | +1.1% | Figure 3 | [`linear_attention/gdn_prefill_bwd/h100`](linear_attention/gdn_prefill_bwd/h100/) |
| Gated DeltaNet prefill，变长 | B200 | Kernel Design Agents（KDA-v0.5） | +5.8% | Figure 4 | [`linear_attention/gdn_prefill_varlen/b200`](linear_attention/gdn_prefill_varlen/b200/) |
| 循环式 Kimi Delta Attention prefill | B200 | CAKE | +0.4% | Figure 4 | [`linear_attention/recurrent_kda_prefill/b200`](linear_attention/recurrent_kda_prefill/b200/) |

### GEMM

| 算子 | GPU | 对比对象 | 加速 | 出处 | 目录 |
|-|-|-|-|-|-|
| 稠密 GEMM，16 位 | H100 | cuBLAS | +0.9% | Figure 3 | [`gemm/gemm_fp16/h100`](gemm/gemm_fp16/h100/) |
| FP8 GEMM，128 块缩放 | H100 | DeepGEMM | +0.8% | Figure 3 | [`gemm/fp8_gemm/h100`](gemm/fp8_gemm/h100/) |
| FP8 × FP4 grouped GEMM（MoE） | B200 | DeepGEMM | +0.6% | Figure 3 | [`gemm/fp4_grouped_gemm/b200`](gemm/fp4_grouped_gemm/b200/) |

### 卷积

| 算子 | GPU | 对比对象 | 加速 | 出处 | 目录 |
|-|-|-|-|-|-|
| Conv2d 权重梯度 | H100 | cuDNN | +1.0% | Figure 3 | [`conv/conv2d_wgrad/h100`](conv/conv2d_wgrad/h100/) |
| Conv2d 数据梯度，stride 2 | H100 | cuDNN | +0.7% | Figure 3 | [`conv/conv2d_dgrad/h100`](conv/conv2d_dgrad/h100/) |
| Conv3d 前向 | H100 | cuDNN | +7.6% | Figure 3 | [`conv/conv3d_fwd/h100`](conv/conv3d_fwd/h100/) |

### 代际

Table 1 的 kernel 生成时，都把当时已知最好的 kernel 封装后作为参照；加速是相对这个参照，括号里是相对专家库的累计加速。
同一算子在同一 GPU 上有多个 kernel 时，分子目录存放：

- `gen0` 是 ForgeKernel 对着专家库生成的第一个 kernel（Figure 3）。
- `gen1`、`gen2` 是连续晋升的结果：每一代生成时都把上一代封装后作为参照。论文里只有 Attention 前向这条链连续晋升了两次，
  所以只有它有 `gen1` 和 `gen2`。
- `from_agent` 只晋升了一次，参照是 /goal agent 写的 kernel，不是 ForgeKernel 的产出，所以参照本身不在本目录里。

## 环境

| GPU | CUDA | PyTorch | nvidia-cutlass-dsl |
|-|-|-|-|
| H100 SXM（`sm_90a`） | 12.8 或 13.1 | 2.6 或 2.10 | 4.4.2、4.6.0.dev0 或 4.6.1 |
| B200（`sm_100a`） | 13.0 | 2.13 | 4.6.0 或 4.6.2（CUDA C++ 那个算子不用） |

计时用的是 device 时间，即连续背靠背发射取中位数；
部分 H100 测量把 SM 时钟锁在 1590 MHz。

## 许可证

Apache License 2.0，见 [LICENSE](../LICENSE) 与 [NOTICE](../NOTICE)。CuTe DSL kernel 依赖 NVIDIA CUTLASS
CuTe DSL（`nvidia-cutlass-dsl`），它不包含在本仓库中，按其自身许可证分发。
