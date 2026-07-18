**AVDM² (Accelerating Video Diffusion Models via Distribution Matching)** arXiv:2412.05899 | Rhymes.AI + Nanjing University | Dec 2024

------

### TL;DR

将预训练视频 diffusion model（AnimateDiff）蒸馏为 **4-step few-step generator**，核心贡献是在视频蒸馏场景下将 **Denoising GAN（ADM）** 与 **2D Score Distribution Matching（SDM）** 联合训练。实验结果 FVD 1271 / CLIPScore 32.01，超过 AnimateDiff-Lightning、AnimateLCM 等 SOTA。

------

### 问题背景

视频 diffusion 推理极慢（高分辨率长视频 > 10 分钟 @ 50 steps），但现有视频蒸馏方案（LCM、Consistency Model）在 4-step 下：

- 帧质量模糊（regression loss 的 mode averaging 问题）
- motion-appearance 不一致
- 语义对齐差（CLIPScore 低）

纯 fast sampler（DDIM、DPM-solver）受 PF-ODE 高曲率限制，< 10 steps 质量断崖。

------

### 方法：双 Distribution Matching 损失

#### 整体目标函数

$$
\mathcal{L}(\theta) := \lambda_{\text{SDM}} \mathcal{L}_{\text{SDM}}(\theta) + \lambda_{\text{ADM}} \mathcal{L}_{\text{ADM}}(\theta)
$$

两个损失作用于不同 distribution 对齐目标。

------

#### Loss 1：Video Adversarial Distribution Matching（ADM）

即 **Denoising GAN**，将 generator 输出分布对齐**真实视频数据分布**。

标准 denoising GAN 目标：
$$
\min_{G_\theta} \max_{D_\eta} \mathbb{E}_{x \sim p_r, t \sim [t_{g}^{\min}, t_{g}^{\max}]}[\log D_\eta(x_t, t)] + \mathbb{E}_{\epsilon \sim p_\epsilon, t}[\log(1 - D_\eta(\tilde{x}_t, t))]
$$
Generator 的 non-saturating loss：
$$
\mathcal{L}_{\text{ADM}}(\theta) = \mathbb{E}_{\epsilon \sim p_\epsilon, t \sim [t_g^{\min}, t_g^{\max}]} \left[ \log D_\eta(\alpha_t G_\theta(\epsilon) + \sigma_t \epsilon', t) \right]
$$
**判别器结构**：冻结 video teacher UNet encoder + 可训练多尺度 prediction head（提取 3 个空间尺度特征 concat 后接 conv 输出 logit）。

> 注：ADM 中 $t $ 加噪是为了让 $p_r $ 与 $p_g $ 的 support 有重叠（解决 GAN 原始 non-overlapping 问题）。

------

#### Loss 2：2D Frame Score Distribution Matching（SDM）

即 VSD（Variational Score Distillation），作用于**逐帧质量**，利用预训练的 **2D image diffusion model** 做监督（而非视频 diffusion），灵活引入更强的 image prior。

SDM gradient：
$$
\nabla_\theta \mathcal{L}_{\text{SDM}} = \mathbb{E}_{t, \epsilon} \, w(t) \left[ \epsilon_\phi(\tilde{x}_t^K, t) - \epsilon_\psi(\tilde{x}_t^K, t) \right] \frac{d G_\theta^K(\epsilon)}{d\theta}
$$
其中：

- $G_\theta^K(\epsilon)$：从 generator 输出视频中随机采 $K $ 帧
- $\epsilon_\phi$：2D teacher diffusion model（固定）
- $\epsilon_\psi$：2D "fake model"，用 generator 输出帧 fine-tune，学习 generator 当前分布的 score function

**关键区分**：SDM 只作用于 **generator 的最终 clean sample**（$\tilde{x}_0^K $），不像 video SDM 作用于整个视频。这让任意共享 latent space 的 2D model 都可以直接插入，而且显存开销小得多（对视频模型蒸馏非常重要）。

**为什么 2D SDM 优于 video SDM**：CLIPScore 32.01 vs 30.56（见 Table 1），语义对齐更强，原因是当前 2D image model 的图像质量先验远强于视频 model。

------

#### Backward Simulation

推理时 generator 是 $t: T \to 0 $ 的 few-step ODE，但训练时 generator 的输入分布和推理有 mismatch。采用 **backward simulation**（先跑一遍 generator 正向生成 $\tilde{x}_0 $，再对 $\tilde{x}_0 $ 做 forward diffusion 得 $\tilde{x}_t $）缓解 train-inference gap。

------

#### 训练流程（Algorithm 1 要点）

| 更新对象                       | 频率 | 目标                                                         |
| ------------------------------ | ---- | ------------------------------------------------------------ |
| Generator $G_\theta $          | 1x   | $\lambda_{\text{SDM}} \mathcal{L}_{\text{SDM}} + \lambda_{\text{ADM}} \mathcal{L}_{\text{ADM}} $ |
| 2D fake model $\epsilon_\psi $ | 2x   | diffusion regression loss（学 generator 当前分布 score）     |
| Discriminator $D_\eta $        | 2x   | GAN objective                                                |

two-timescale update（判别器和 fake model 更新频率是 generator 的 2 倍）稳定训练。

------

### 实验结果（AnimateDiff teacher，4 steps）

| 方法                     | FVD ↓       | CLIPScore ↑ |
| ------------------------ | ----------- | ----------- |
| Motion Consistency Model | 1765.30     | 28.60       |
| AnimateLCM               | 1405.79     | 28.44       |
| AnimateDiff-Lightning    | 1623.98     | 29.47       |
| **Ours (2D SDM)**        | **1271.45** | **32.01**   |

消融实验验证："GAN alone 不够"，SDM 是提升 prompt following 和帧质量的关键。

