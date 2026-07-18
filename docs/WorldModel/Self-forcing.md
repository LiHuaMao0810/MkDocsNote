# Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion

**arXiv:2506.08009 | NeurIPS 2025 Spotlight | Adobe Research & UT Austin**

------

## 1. 核心问题：Exposure Bias

### 背景

AR Video Diffusion 的联合分布分解为：

$$p(x^{1:N}) = \prod_{i=1}^{N} p(x^i | x^{<i})$$

每个条件分布 $p(x^i | x^{<i})$ 用 diffusion process 建模，通过逐步去噪生成 $x^i$。

标准训练用 **Teacher Forcing (TF)** 或 **Diffusion Forcing (DF)**：

$$\mathcal{L}^{\text{DM}}_\theta = \mathbb{E}_{x^i, t^i, \epsilon^i}\left[w_{t^i} \|\hat{\epsilon}^i_\theta - \epsilon^i\|_2^2\right]$$

其中 context 是 **ground-truth 帧** $x^{<i}$（TF）或 **加噪 GT 帧** $x^{j<i}_{t^j}$（DF）。

### 问题所在

训练时用 GT context → 推理时用自己生成的（有缺陷的）帧作 context，形成 **train-test distribution mismatch**，误差随帧数累积。

CausVid（最接近的 prior work）用 DF + DMD 蒸馏，但其训练生成样本来自 DF 分布，**不等于推理时模型的真实分布**，因此 DMD loss 匹配的是错误的分布。

------

## 2. 方法：Self Forcing	

### 核心思路

**训练时执行 autoregressive self-rollout**：每帧的 context 不是 GT，而是模型自己之前生成的帧，完全模拟推理过程。

$${x^{1:N}_\theta} \sim p_\theta(x^{1:N}) = \prod_{i=1}^{N} p_\theta(x^i | x^{<i}_\theta)$$

训练时用 **KV caching**（通常只在推理时用），让每帧的 attention 直接 attend 到之前自生成帧的 KV embeddings。

### 训练目标（两部分）

**（1）Holistic Video-Level Distribution Matching Loss**

由于 self-rollout 后得到的是来自真实模型分布 $p_\theta$ 的样本，可以直接用 video-level 分布匹配 loss：

论文支持三种可插拔的 loss：

| Loss                                         | 形式                                                         | 参考      |
| -------------------------------------------- | ------------------------------------------------------------ | --------- |
| **SiD** (Score identity Distillation)        | $\mathcal{L}_\text{SiD} = \mathbb{E}[\|s_\theta(x) - s_\text{real}(x)\|^2]$ | SiD paper |
| **DMD** (Distribution Matching Distillation) | $\mathcal{L}_\text{DMD} = \mathbb{E}_{x \sim p_\theta}[s_\theta(x) - s_\text{real}(x)]$ | DMD/DMD2  |
| **GAN**                                      | $\mathcal{L}_\text{GAN} = \mathbb{E}_{x \sim p_\theta}[\log D(x)] + \mathbb{E}_{x \sim p_\text{real}}[\log(1-D(x))]$ | -         |

这些 loss 作用在**完整生成视频**上，而非逐帧计算，因此能捕捉跨帧一致性。

**（2）Stochastic Gradient Truncation（关键效率设计）**

​	若对每帧的完整多步去噪链 backprop，内存爆炸。设计：

- 使用 **few-step diffusion backbone**（$T$ 步，$T$ 较小，~4步）
- 每次训练：sample $s \sim \text{Uniform}(1, \ldots, T)$，**只对第 $s$ 步的输出计算 loss + backprop**（随机截断）
- 对 **KV cache embeddings 的梯度 detach**（不跨帧 backprop），避免显存随帧数线性增长

这样每步只需 backprop 通过单次 denoising step，所有 denoising steps 均有机会收到梯度信号。

**整体 training loop（Algorithm 1 概述）：**

```
for each training step:
    sample s ~ Uniform(1, ..., T)  # 随机截断位置
    KV = []
    Xθ = []
    for i = 1..N:  # 逐帧 AR rollout
        sample xtT^i ~ N(0,I)
        for j = T, T-1, ..., 1:
            xtj-1^i = f_θ(xtj^i, x<i_θ via KV)  # few-step denoising
        xθ^i = xt0^i
        if j == s:
            compute distribution matching loss on xθ^i
        KV.append(KV embeddings of xθ^i, detached)
        Xθ.append(xθ^i)
    backprop only through step s of current frame
```

------

## 3. Rolling KV Cache（长视频外推）

为支持任意长视频生成，提出 **Rolling KV Cache**：当 context 帧超出固定窗口时，丢弃最旧的帧 KV，保留最近 $W$ 帧。

训练时加入 **local attention mask** 训练，让模型学会只依赖局部 context，否则外推时会出现分布偏移。

------

## 4. 与 Prior Work 的核心对比

|                | Teacher Forcing        | Diffusion Forcing (CausVid) | **Self Forcing**                        |
| -------------- | ---------------------- | --------------------------- | --------------------------------------- |
| 训练 context   | GT clean frames        | GT noisy frames             | **自生成帧**                            |
| 推理时分布匹配 | ✗（GT ≠ 推理分布）     | ✗（noisy GT ≠ 推理分布）    | **✓（完全一致）**                       |
| Attention mask | 特殊 block-sparse mask | 特殊 causal mask            | **无需特殊 mask（与推理完全相同）**     |
| KV cache       | 仅推理用               | 仅推理用                    | **训练推理都用**                        |
| Loss level     | 逐帧 MSE               | 逐帧 MSE + DMD（分布不准）  | **Video-level distribution matching**   |
| 并行性         | 全并行                 | 全并行                      | 顺序（但 per-frame 内部并行，效率仍高） |

**对 CausVid 的精确定位批评**：CausVid 的 DMD loss 用 DF 生成的样本（含 noisy context），但推理时 context 是 clean 自生成帧，两者分布不同 → DMD 优化目标与实际推理分布脱节，导致 over-saturation artifacts。

------

## 5. 实验结果

- **速度**：单卡 H100 实时 17 FPS，首帧延迟 ~0.8s；比 Wan/SkyReels/MAGI 快 **150–400×**
- **质量**：VBench 上达到或超过 Wan2.1-1.3B、SkyReels2-1.3B、MAGI-1-4.5B
- **同速比质量**：相同速度下（vs CausVid），无 over-saturation，motion 更自然
- **数据效率**：training is data-free（从 CausVid 的 ODE 初始化 checkpoint 出发做 post-training，不需要视频数据）