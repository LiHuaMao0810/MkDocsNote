

## Self-Forcing++：Towards Minute-Scale High-Quality Video Generation

**arXiv:2510.02283 | Oct 2025**

------

### TL;DR

核心问题是**长视频自回归生成中的训练-推理分布偏移（Exposure Bias）**。当 student 模型生成的视频超出 teacher 模型的训练时域时，误差在连续 latent space 中不断累积，导致画面过曝、色彩漂移、最终质量崩塌。

Self-Forcing++ 的核心洞见：**用 student 自己生成的长视频中采样的片段，反过来让 teacher 模型对这些片段提供监督**，从而在不需要长视频 GT 数据、也不需要重新训练的前提下，将生成时域外推到 teacher 能力的 20×，最长可达 **4 分 15 秒**（约 255 秒）。

------

### 背景：Autoregressive Video Diffusion 的演化脉络

理解 Self-Forcing++ 需要先理清其前置工作的递进逻辑：

| 范式                                                    | 训练时 Context 来源                                  | 问题                                                         |
| ------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| **Teacher Forcing**                                     | 干净 GT 帧                                           | 推理时 context 是自生成帧，分布不匹配                        |
| **Diffusion Forcing** (Chen et al., NeurIPS 2024)       | 每帧独立噪声级别                                     | 缓解了 TF 的不匹配，但 DMD loss 匹配分布仍有缺陷             |
| **CausVid**                                             | DF scheme + Distribution Matching Distillation (DMD) | DMD 匹配的是 DF 输出分布而非真实推理分布，长时 drift 仍存在  |
| **Self-Forcing** (Huang et al., NeurIPS 2025 Spotlight) | 训练时做真实自回归 rollout                           | 解决了 train-test gap，但生成时域被限制在 teacher horizon 内 |
| **Self-Forcing++** (Cui et al., 2025)                   | 自生成长视频中随机采样片段                           | 突破 teacher horizon，实现分钟级生成                         |

------

### 核心问题定位

设 teacher 模型的最大生成时域为 $T_\text{teacher}$（典型值 5-10 秒），student 进行自回归外推到 $T > T_\text{teacher}$ 时，有两类误差来源：

1. **Exposure Bias**：训练时 student 看到干净上下文，推理时条件于自己生成的有噪帧，分布偏移导致误差逐帧放大。
2. **Long-horizon Underdetermination**：teacher 从未见过 $T > T_\text{teacher}$ 的视频，因此对 student 在这段区域的生成无法提供有效监督，梯度信号为零，导致质量崩塌（over-exposure、color drift）。

------

### 方法：Self-Forcing++

**核心思路**：student 自己做长视频 rollout（$T \gg T_\text{teacher}$），然后从这条长轨迹上随机采样一个短片段（长度 $\approx T_\text{teacher}$），让 teacher 对这个片段计算 distribution matching loss，提供监督梯度。

训练流程（概念伪代码）：

```python
# Phase 1: Student 自回归生成长视频（无梯度）
with torch.no_grad():
    long_video = student_autoregressive_rollout(
        model=student,
        length=T_long,  # >> T_teacher
        kv_cache=True
    )  # shape: (B, T_long, C, H, W)

# Phase 2: 从长视频中随机采样短片段
t_start = random.randint(0, T_long - T_teacher)
segment = long_video[:, t_start:t_start + T_teacher]

# Phase 3: Teacher 对采样片段计算 distribution matching loss
# （student 对这个 segment 做有梯度的 re-denoise）
loss = distribution_matching_loss(
    teacher=teacher,
    student=student,
    segment=segment,
    noise_init="backward_noise"  # 见下方设计细节
)
loss.backward()
optimizer.step()
```

这样做的关键优势：

- Teacher 监督信号覆盖了 $[0, T_\text{long}]$ 的任意时间位置，**而不只是前 $T_\text{teacher}$ 帧**
- 无需任何长视频 GT 数据
- 无需重训 teacher

------

### 关键设计：Backward Noise Initialization

Self-Forcing++ 使用 **backward noise initialization** 和 rolling KV cache 机制来实现高效的时序外推，避免了前置方法中重计算 overlapping frames 的开销。

Backward Noise Initialization 的直觉：当对 long video 中的某个片段重新施加噪声时，使用与前向生成过程一致的噪声初始化方向，保证 distribution matching 的分布是一致的，而非随机重噪引入额外偏差。

------

### 训练目标

Distribution Matching Distillation 的 loss 形式（继承自 CausVid/Self-Forcing 脉络，Self-Forcing++ 中作用于自生成片段）：

$$\mathcal{L}*\text{DMD} = \mathbb{E}*{t, \epsilon}\left[ w(t) \cdot \left| \hat{\epsilon}*\text{student}(x_t^\text{seg}) - \hat{\epsilon}*\text{teacher}(x_t^\text{seg}) \right|^2 \right]$$

其中 $x_t^\text{seg}$ 是从 student 生成长视频中采样的片段加噪后的版本，$w(t)$ 是 SNR 相关权重。这本质上是让 student 在长视频的任意时间切片上的 score 与 teacher 对齐。

------

### 核心改进 vs. Baselines

| 维度                     | CausVid                        | Self-Forcing                | Self-Forcing++                  |
| ------------------------ | ------------------------------ | --------------------------- | ------------------------------- |
| 训练时 context           | GT（Diffusion Forcing scheme） | 自生成 rollout              | 自生成长 rollout                |
| DMD 分布是否匹配         | ❌（匹配 DF 分布而非推理分布）  | ✅                           | ✅                               |
| 长时域外推               | ❌（超出 teacher 后崩塌）       | ❌（受限于 teacher horizon） | ✅（最长 255 秒）                |
| 是否需要长视频数据       | ❌                              | ❌                           | ❌                               |
| Overlapping frame 重计算 | ✅（有开销）                    | ✅（有开销）                 | ❌（通过 rolling KV cache 消除） |

------

### 实验结果

Self-Forcing++ 在 teacher 能力边界之外将视频长度外推了最高 20 倍，避免了过曝和误差累积问题，同时不需要重计算 overlapping 帧。

视觉稳定性（Visual Stability Score）从 baseline 方法的约 40 跃升至 Self-Forcing++ 的 90 以上，且在所有评测长度（75s、100s）上均显著优于 baseline。

扩展计算预算后，生成能力最高可达 4 分 15 秒，相当于 baseline 模型的 50 倍以上