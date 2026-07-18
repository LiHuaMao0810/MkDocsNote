---
tags:
  - 流匹配

---

# Mean Flow

> [!INFO] 文档信息
>
> 创建时间：2025-12-19 | 更新时间：2026-4-19
>
> 原文链接**[Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447v1)** 

### TL;DR

提出 **average velocity** 场 $u(z_t, r, t)$ 替代 Flow Matching 的 instantaneous velocity $v(z_t, t)$，从定义出发严格推导训练目标（MeanFlow Identity），无需蒸馏/预训练/课程学习，1-NFE 在 ImageNet 256×256 达到 FID 3.43，比此前 SOTA Shortcut 的 10.60 提升约 70%。

------

### 核心概念：Average Velocity

Flow Matching 建模的是**瞬时速度** $v(z_t, t)$，采样时需要数值积分（多步 ODE）。

MeanFlow 定义**平均速度**：

$$u(z_t, r, t) \triangleq \frac{1}{t - r} \int_r^t v(z_\tau, \tau) d\tau$$

**采样**只需一次网络调用：

$$z_r = z_t - (t - r) u_\theta(z_t, r, t)$$

1-step：$z_0 = \epsilon - u_\theta(\epsilon, 0, 1)$，其中 $\epsilon \sim \mathcal{N}(0,I)$。

------

### 训练目标推导：MeanFlow Identity

从定义式 $(t-r)u = \int_r^t v d\tau$ 两边对 $t$ 求全导数（$r$ 视为常数），用乘积法则 + 微积分基本定理：

$$\boxed{u(z_t, r, t) = v(z_t, t) - (t - r)\frac{d}{dt}u(z_t, r, t)}$$

其中全导数展开（链式法则，$\dot{z}_t = v$）：

$$\frac{d}{dt}u = v(z_t, t)\partial_z u + \partial_t u \quad \leftarrow \text{即 JVP}(u;(v, 0, 1))$$

------

### 损失函数

$$\mathcal{L}(\theta) = \mathbb{E}\left| u_\theta(z_t, r, t) - \operatorname{sg}(u_{\text{tgt}}) \right|^2$$

$$u_{\text{tgt}} = v_t - (t - r)\underbrace{\left(v_t\partial_z u_\theta + \partial_t u_\theta\right)}_{\text{jvp}(u_\theta,(z,r,t),(v,0,1))}$$

其中 $v_t = \epsilon - x$ 为 conditional velocity（替代不可算的 marginal velocity），$\operatorname{sg}$ 为 stop-gradient（避免二阶梯度）。

**当 $r = t$ 时**，第二项消失，退化为标准 CFM。

------

### 训练流程（伪代码）

```python
t, r = sample_t_r()               # lognorm(-0.4, 1.0)，25%概率 r≠t
e = randn_like(x)
z = (1 - t) * x + t * e           # interpolant
v = e - x                         # conditional velocity
u, dudt = jvp(fn, (z, r, t), (v, 0, 1))   # 单次 backward
u_tgt = v - (t - r) * dudt
loss = metric(u - stopgrad(u_tgt))  # adaptive weight, p=1.0
```

JVP 开销仅约 16%（对比纯 FM），因为 `dudt` 被 stop-gradient，不参与 $\theta$-backprop。

------

### CFG 集成（无额外 NFE）

将 CFG 速度场定义在 ground-truth 层面：

$$v^{\text{cfg}}(z_t, t \mid c) \triangleq \omega v(z_t, t \mid c) + (1 - \omega)v(z_t, t)$$

对应的 $u^{\text{cfg}}$ 满足同样的 MeanFlow Identity，训练目标中将 $v_t$ 替换为：

$$\tilde{v}*t = \omega, v_t + (1-\omega), u_\theta^{\text{cfg}}(z_t, t, t)$$

采样时直接用 $u_\theta^{\text{cfg}}(\epsilon, 0, 1)$，保持 1-NFE。

------

### 与相关工作的核心区别

| 方法               | 时间变量                 | 约束来源       | 是否需要预训练 |
| ------------------ | ------------------------ | -------------- | -------------- |
| Consistency Models | 单变量 $t$（固定 $r=0$） | 网络行为约束   | CT需要；CD需要 |
| Shortcut / IMM     | 双变量 $(r,t)$           | 额外自洽性损失 | 需要           |
| **MeanFlow**       | 双变量 $(r,t)$           | 定义直接推导   | **不需要**     |

MeanFlow Identity 是 $u$ 定义的**必要且充分条件**（Appendix B.3），不依赖网络结构假设。

------

### 关键超参数（ImageNet XL/2 默认）

| 超参                | 取值                                      |
| ------------------- | ----------------------------------------- |
| $(r,t)$ 采样        | lognorm$(-0.4, 1.0)$，$r \neq t$ 占 25%   |
| 网络条件输入        | $(t,, t-r)$ 位置编码                      |
| 自适应 loss weight  | $w = 1/(|\Delta|^2 + 10^{-3})^p$，$p=1.0$ |
| CFG scale $\omega'$ | 2.0（XL/2）                               |
| EMA decay           | 0.9999                                    |

------

### 与你的 GRPO+FM 研究的接口

MeanFlow 的 $u_\theta(\epsilon, 0, 1)$ 本质上是一个**确定性的 one-step generator**，可直接作为你 GRPO 框架中的 student policy：reward 对 $u_\theta$ 求梯度，IS ratio 在 one-step 情形下退化为单点比值，理论上比多步场景更简洁。值得关注。