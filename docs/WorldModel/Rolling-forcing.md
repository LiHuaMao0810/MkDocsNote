# Rolling Forcing

### TL;DR

**一句话定位**：针对 autoregressive 流式长视频生成中的 **error accumulation** 和 **exposure bias** 问题，提出三个互补设计，在单 GPU 上实现实时（16 fps）、多分钟级别的流式视频生成。

**Base model**：Wan2.1（开源）。Post-training only，仅需 3000 steps，无需视频数据。

------

### 问题背景

#### 两类现有范式的局限

| 范式                | 代表                  | 问题                                                         |
| ------------------- | --------------------- | ------------------------------------------------------------ |
| History Corruption  | Chen 2024, Guo 2025   | 往历史帧注噪→降 exposure bias，但破坏 temporal consistency（clean reference 丢失） |
| Planning Generation | Zhang & Agrawala 2025 | 先生成远端 key frames 再插值→缓解 drift，但 out-of-order 发帧，无法 streaming |
| Causal Distillation | CausVid, Self-Forcing | 支持实时 streaming，但 strict causality → 误差逐帧累积，长视频严重退化 |

**核心矛盾**：strict causality（streaming 必须）vs. error accumulation suppression（长视频必须）。

#### Exposure Bias 的形式化

训练时条件是 ground-truth 历史帧 $\mathbf{x}^0_{<t}$，推理时条件是自身生成的历史 $\hat{\mathbf{x}}_{<t}^0$：
$$
\mathcal{L}_{\text{exposure bias}} = d\!\left(p_\theta(\mathbf{x}_t \mid \mathbf{x}_{<t}^0),\; p_\theta(\mathbf{x}_t \mid \hat{\mathbf{x}}_{<t}^0)\right)
$$
两者分布 mismatch 随步数指数级放大，即经典 AR 模型的 compounding error 问题。

------

### 三个核心设计

#### 设计 1：Rolling-Window Joint Denoising

**关键思路**：在一个滑动窗口内同时去噪 $W $ 帧，各帧分配**递增的噪声级别**：
$$
\mathbf{x}_{t}^{(\sigma_0)}, \mathbf{x}_{t+1}^{(\sigma_1)}, \ldots, \mathbf{x}_{t+W-1}^{(\sigma_{W-1})}, \quad \sigma_0 < \sigma_1 < \cdots < \sigma_{W-1}
$$
其中最左帧噪声最小（最干净），最右帧噪声最大（最"未来"）。窗口内用 **bidirectional attention** 连接所有帧，使右侧较脏的帧能向左侧干净帧"借助"上下文进行互相修正（mutual refinement）。

每次 forward pass 后，最左帧已足够干净即可 **emit**，窗口向右滑动一帧。

> 对比 Rolling Diffusion（Ruhe et al. 2024）：本文在此基础上加入了 post-training distillation 和 exposure bias 修复，原始 Rolling Diffusion 的 exposure bias 问题依然存在。

**实时性保障**：尽管 attention window 扩大，但每次 forward 仍然只 emit 1 帧，throughput 可维持 16 fps。

------

#### 设计 2：Attention Sink for Global Consistency

借鉴 StreamingLLM（Xiao et al. 2023）的 attention sink 思想，将初始帧的 **KV states 永久保留**作为全局 anchor：

- 初始帧 KV cache 不随滑动窗口淘汰（普通 KV cache 只保留近期帧）
- 动态调整初始帧的 **RoPE 位置编码**：将其相对当前去噪帧的位置固定（freezing relative position），避免 position offset 过大导致 attention weight 退化

$$
\text{RoPE}_{\text{sink}} \leftarrow \text{RoPE}(0 \to \text{current frame index offset})
$$

效果：长视频生成中全局光照/场景一致性显著改善，ablation 中去掉 attention sink 后 temporal quality 大幅下降。

------

#### 设计 3：Rolling Forcing Post-Training（Distillation + Exposure Bias Fix）

**训练目标**：在 **non-overlapping windows** 上做 few-step distillation，同时 condition on **自身生成的历史帧**（而非 GT），从而在训练阶段主动引入 exposure bias，使模型学会在 imperfect history 下仍能生成高质量帧。

训练流程（混合策略，ablation 中称 "mixed training"）：

1. **Self Forcing training（SF training）**：条件为自生成历史，计算 consistency distillation loss，mitigate exposure bias
2. **Rolling Forcing training（RF training）**：在完整 rolling window 上训练，学习窗口内的 joint denoising 行为

**Loss 设计**：few-step consistency distillation（类 CausVid），teacher 为预训练 Wan2.1：
$$
\mathcal{L} = \mathbb{E}\left[\left\|\hat{\mathbf{x}}_\theta(\mathbf{z}_{t_k}, t_k, \hat{h}) - \mathbf{x}_{\text{teacher}}(\mathbf{z}_{t_k}, t_k, h)\right\|^2\right]
$$
其中 $\hat{h} $ 是 student 自生成历史，$h $ 是 GT 历史（teacher 用 GT，student 用 self-generated，是 Self Forcing 思想的直接应用）。

**窗口设计**：用 non-overlapping window 而非 overlapping，原因是 overlapping 会导致同一帧在多个 window 中被重复监督，梯度方向不一致；non-overlapping 保证每帧只被监督一次，训练更稳定。

------

### 实验结论

**VBench 定量**（vs. CausVid、Self-Forcing 等 baseline）：

| 方法                | Temp. Consist. | Motion    | Aesthetic | Error Accum.     |
| ------------------- | -------------- | --------- | --------- | ---------------- |
| Self Forcing        | 较高           | -         | -         | 严重             |
| **Rolling Forcing** | **97.61**      | **98.70** | **70.75** | **极低（0.01）** |

- 单 GPU 上实现实时流式生成，能产出多分钟级视频，error accumulation 大幅降低。 [Hugging Face](https://huggingface.co/papers/2509.25161)
- 仅训练 3000 步，不需要视频数据（只需图像/短片段即可），对比 StreamDiT 需要大规模预训练数据，成本优势明显。 [arxiv](https://arxiv.org/html/2509.25161v1)
- Ablation：去掉 RF inference → error accum 指标从 0.01 升至 5.53；去掉 attention sink → 时序一致性显著下降。

------

### 关键创新点 vs. 最近同类工作

| 对比项             | Self-Forcing | CausVid | Rolling Diffusion | **Rolling Forcing** |
| ------------------ | ------------ | ------- | ----------------- | ------------------- |
| 是否 streaming     | ✓            | ✓       | ✓                 | ✓                   |
| Error Accum. 抑制  | ✗            | ✗       | 部分              | ✓（核心贡献）       |
| Exposure Bias Fix  | ✓            | ✗       | ✗                 | ✓                   |
| Global Consistency | ✗            | ✗       | ✗                 | ✓（attention sink） |
| 无需修改架构       | ✓            | ✓       | ✗                 | ✓                   |
| 无需视频训练数据   | 部分         | 部分    | -                 | ✓                   |