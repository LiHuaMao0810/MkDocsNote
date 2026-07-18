---
tags:
  - 分数匹配
  - 蒸馏
  - 扩散模型
---

# **DMD 分布匹配蒸馏**

> [!info]
>
> 创建时间：2025-11-29 | 更新时间：2026-4-19
>
> 原文链接**[One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828)** 

### TL;DR

将预训练多步扩散模型蒸馏为**单步生成器**，核心创新是用 **KL 散度梯度**（real score − fake score 之差）作为分布匹配信号，而非直接监督 noise→image 映射。ImageNet 64×64 达到 FID=2.62，SD 蒸馏后 FID=11.49，速度提升 30×。

------

### 问题设定

给定预训练扩散模型 $\mu_\text{base}$，训练单步生成器 $G_\theta: \mathbb{R}^d \to \mathbb{R}^d$，使：

$$z \sim \mathcal{N}(0, I) \xrightarrow{G_\theta} x \sim p_\text{fake} \approx p_\text{real}$$

------

### 训练目标

#### 1. Distribution Matching Loss（核心）

最小化 $D_\text{KL}(p_\text{fake} \| p_\text{real})$，其梯度为：

$$\nabla_\theta D_\text{KL} = \mathbb{E}_{z, x=G_\theta(z)} \left[ -\left(s_\text{real}(x) - s_\text{fake}(x)\right) \frac{dG}{d\theta} \right]$$

**两个难点及解决方案：**

- $p_\text{real}$ 在 fake 样本处趋于零 → 对生成图像加噪 $x_t \sim q_t(x_t|x) = \mathcal{N}(\alpha_t x,\ \sigma_t^2 I)$，使分布有全局支撑
- Score 难以直接计算 → 用两个扩散模型分别参数化 real/fake score

加噪后的近似梯度：

$$\nabla_\theta D_\text{KL} \simeq \mathbb{E}_{z, t, x, x_t} \left[ w_t \alpha_t \left( s_\text{fake}(x_t, t) - s_\text{real}(x_t, t) \right) \frac{dG}{d\theta} \right]$$

其中 score 由 EDM mean-prediction 形式给出：

$$s_\text{real}(x_t, t) = -\frac{x_t - \alpha_t \mu_\text{base}(x_t, t)}{\sigma_t^2}, \quad s_\text{fake}(x_t, t) = -\frac{x_t - \alpha_t \mu^\phi_\text{fake}(x_t, t)}{\sigma_t^2}$$

**梯度方向的直觉：** $s_\text{real}$ 把 $x$ 推向真实分布的 mode，$-s_\text{fake}$ 阻止 $x$ 聚集成高 fake density 区域，二者之差同时增大 realism 并防止 mode collapse。

#### 2. Fake Score 动态更新

$\mu^\phi_\text{fake}$ 随 $p_\text{fake}$ 变化而持续训练：

$$\mathcal{L}^\phi_\text{denoise} = | \mu^\phi_\text{fake}(x_t, t) - x_0 |_2^2, \quad x_0 = \text{stopgrad}(G_\theta(z))$$

注意 $x_0$ 需 stop gradient，防止 $\mu^\phi_\text{fake}$ 反过来影响 $G_\theta$。

#### 3. Regression Loss（防 mode collapse）

离线预生成 paired dataset $\mathcal{D} = {(z, y)}$（ODE solver 多步采样），然后：

$$\mathcal{L}_\text{reg} = \mathbb{E}_{(z,y)\sim\mathcal{D}}\ \text{LPIPS}(G_\theta(z),\ y)$$

纯分布匹配在低噪声 $t \approx 0$ 时 $s_\text{real}$ 不可靠，且 score 对密度缩放不敏感，会导致 mode dropping。Regression loss 固定 $G_\theta$ 与 teacher 的 pointwise 对应关系。

#### 4. 时间步权重 $w_t$

$$w_t = \frac{\sigma_t^2}{\alpha_t} \cdot \frac{CS}{|\mu_\text{base}(x_t, t) - x|_1}$$

归一化不同噪声水平下的梯度幅度，优于 DreamFusion 的 $\sigma_t/\alpha_t$ 方案（CIFAR FID 改进约 0.9）。

#### 5. 最终目标

$$\min_\theta\ D_\text{KL} + \lambda_\text{reg} \mathcal{L}_\text{reg}, \quad \lambda_\text{reg} = 0.25$$

两个损失分别作用于**不同数据流**：DM loss 用 unpaired fake 样本，Regression loss 用 paired dataset。

------

### 训练流程（Algorithm 1 精要）

```
每个 iteration:
  Phase A - Update G_θ:
    z ~ N(0,I), x = G_θ(z)
    t ~ U(T_min, T_max),  x_t = forward_diffuse(x, t)
    L_KL = w_t * α_t * (μ_fake(x_t,t) - μ_real(x_t,t)) / norm  # stopgrad on denoisers
    L_reg = LPIPS(G_θ(z_ref), y_ref)
    update G_θ with L_KL + 0.25 * L_reg

  Phase B - Update μ_fake:
    x_t = forward_diffuse(stopgrad(x), t)
    L_denoise = ||μ_fake(x_t,t) - stopgrad(x)||²
    update μ_fake with L_denoise
```

------

### 与相关工作的核心区别

| 方法                   | 蒸馏目标           | 是否需要 paired ODE trajectories | 分布匹配方式                 |
| ---------------------- | ------------------ | -------------------------------- | ---------------------------- |
| PD / CD                | ODE 轨迹一致性     | 是（online）                     | 无                           |
| InstaFlow              | 直化 flow          | 是（offline）                    | 无                           |
| VSD（ProlificDreamer） | Score 蒸馏         | 否                               | 有，但用于 test-time 优化 3D |
| **DMD**                | KL 散度（score差） | 是（offline，少量）              | 有，用于训练生成器           |

DMD 的关键洞察：offline 预生成 paired data 足够作为 regression 正则，一旦加入 distribution matching gradient，regression-only baseline（FID 9.21）就跳升至 2.62。
