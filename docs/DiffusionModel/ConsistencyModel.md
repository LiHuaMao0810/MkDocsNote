---
tags:
  - 蒸馏
---

# Consistency Model 一致性模型

>[!INFO] 文档信息
>
>创建时间：2025-12-19 | 更新时间：2026-3-29
>
>原文链接**[Consistency Models](https://arxiv.org/abs/2303.01469)** 

## TL;DR

**问题**：Diffusion Model 的 PF ODE 轨迹上的点与 $x_0$ 存在确定性映射，但采样需要多次 ODE solver 调用（10–2000 NFE）。

**核心贡献**：提出**一致性函数**的概念，训练一个网络直接学习"轨迹上任意点 → 轨迹起点 $x_\epsilon$"的映射，实现单步生成，同时保留多步采样 quality-compute tradeoff。

**vs Baseline（Progressive Distillation, PD）**：

- CD（Consistency Distillation）在 CIFAR-10 上 1-step FID: **3.55 vs 8.34**（PD）
- 无需构造合成数据集，直接对相邻时间步的输出做 consistency loss

------

## 1. 前置：Diffusion 的 PF ODE

采用 EDM（Karras et al., 2022）的参数化：$\mu(x,t)=0$，$\sigma(t)=\sqrt{2t}$，前向 SDE：

$$dx_t = \sqrt{2t}, dw_t$$

对应的 **Probability Flow ODE**：

$$\frac{dx_t}{dt} = -t \cdot \nabla \log p_t(x_t)$$

用训练好的 score model $s_\phi(x,t) \approx \nabla \log p_t(x)$ 代入，得到 **empirical PF ODE**：

$$\frac{dx_t}{dt} = -t \cdot s_\phi(x_t, t) \tag{3}$$

采样时：从 $\hat{x}*T \sim \mathcal{N}(0, T^2 I)$ 出发，用 ODE solver 反向积分到 $t=\epsilon$ 得到 $\hat{x}*\epsilon \approx x_0$。

> 超参数设置：$T=80$，$\epsilon=0.002$，像素值归一化到 $[-1,1]$。

------

## 2. Consistency Function 定义

**Definition**：给定 PF ODE 的一条解轨迹 ${x_t}_{t\in[\epsilon,T]}$，定义**一致性函数**：

$$f: (x_t, t) \mapsto x_\epsilon$$

满足 **self-consistency**：同一轨迹上任意两点 $(x_t, t)$ 和 $(x_{t'}, t')$ 满足：

$$f(x_t, t) = f(x_{t'}, t'), \quad \forall t, t' \in [\epsilon, T]$$

**边界条件（Boundary Condition）**：

$$f(x_\epsilon, \epsilon) = x_\epsilon \quad \text{（在 } t=\epsilon \text{ 处为恒等映射）}$$

------

## 3. 网络参数化

训练 consistency model 需要满足：

$$f_\theta(x_\epsilon, \epsilon) = x_\epsilon$$

即在 $t = \epsilon$（最接近数据端）时，模型输出必须等于输入本身（恒等映射）。

如果直接用神经网络 $F_\theta(x, t)$ 来拟合，**没有任何机制保证**它在 $t=\epsilon$ 时输出 $x$，需要额外的 loss 约束，训练不稳定。

------

**Skip Connection 参数化的解法**

把模型设计成：

$$f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot F_\theta(x, t)$$

其中 $c_{\text{skip}}(t)$ 和 $c_{\text{out}}(t)$ 是关于 $t$ 的**固定可微函数**（不是学习参数），满足：

$$c_{\text{skip}}(\epsilon) = 1, \quad c_{\text{out}}(\epsilon) = 0$$

这样在 $t = \epsilon$ 时：

$$f_\theta(x, \epsilon) = 1 \cdot x + 0 \cdot F_\theta(x, \epsilon) = x \quad \checkmark$$

**边界条件自动满足，无论 $F_\theta$ 输出什么。**

可以理解为一个**加权融合**：

- $c_{\text{skip}}(t)$：直接抄输入 $x$ 的比例
- $c_{\text{out}}(t)$：依赖神经网络预测的比例

在 $t=\epsilon$（几乎是干净图像）时，完全信任输入本身；在 $t$ 较大（严重加噪）时，更多依赖网络的预测来"还原" $x_0$。

------

**具体函数形式**

论文沿用 EDM（Karras et al., 2022）的 preconditioning，一个典型选择是：

$$c_{\text{skip}}(t) = \frac{\sigma_{\text{data}}^2}{(t - \epsilon)^2 + \sigma_{\text{data}}^2}, \quad c_{\text{out}}(t) = \frac{\sigma_{\text{data}} \cdot (t - \epsilon)}{\sqrt{\sigma_{\text{data}}^2 + t^2}}$$

这两个函数在 $t \to \epsilon$ 时自然趋近于 $1$ 和 $0$，在 $t$ 大时 $c_{\text{out}}$ 占主导。

因为 $c_{\text{skip}}(t) \cdot x$ 这一项是把输入 $x$ **绕过神经网络 $F_\theta$ 直接加到输出上**，形式上和 ResNet 的 $y = F(x) + x$ 一模一样，所以叫 skip connection，只不过这里的"跳过权重"是关于 $t$ 的函数而不是常数 1。

------

## 4. 两种训练方式

直接从论文原文的 Algorithm 2 和 Algorithm 3 出发，逐步拆解。

###  4.1 Consistency Distillation (CD)

在 PF ODE 轨迹上，用预训练 score model 走**一步 ODE solver**，得到相邻点对 $(x_{t_{n+1}}, \hat{x}^\phi_{t_n})$，强迫 $f_\theta$ 对这两个点输出一致。

**Step 1：时间步离散化**

将 $[\epsilon, T]$ 离散为 $N-1$ 段，边界点 $t_1 = \epsilon < t_2 < \cdots < t_N = T$，按 Karras et al. 的公式：

$$t_i = \left(\epsilon^{1/\rho} + \frac{i-1}{N-1}(T^{1/\rho} - \epsilon^{1/\rho})\right)^\rho, \quad \rho = 7$$

**Step 2：构造相邻点对**

给定数据点 $x \sim p_{\text{data}}$，随机采样时间步索引 $n$：

$$x_{t_{n+1}} \sim \mathcal{N}(x,; t_{n+1}^2 I)$$

用 Heun（2阶）ODE solver 走一步得到 $\hat{x}^\phi_{t_n}$。以 Euler（简化）为例：

$$\hat{x}^\phi_{t_n} = x_{t_{n+1}} - (t_n - t_{n+1}) \cdot t_{n+1} \cdot s_\phi(x_{t_{n+1}}, t_{n+1})$$

$(x_{t_{n+1}}, \hat{x}^\phi_{t_n})$ 是同一条 ODE 轨迹上的相邻两点，理论上应满足 $f(x_{t_{n+1}}, t_{n+1}) = f(\hat{x}^\phi_{t_n}, t_n)$。

**Step 3：计算 CD Loss**

$$\mathcal{L}_{\text{CD}}^N(\theta, \theta^-; \phi) = \mathbb{E}\left[\lambda(t_n) \cdot d\left(f_\theta(x_{t_{n+1}}, t_{n+1}); f_{\theta^-}(\hat{x}^\phi_{t_n}, t_n)\right)\right]$$

注意：

- **online network** $f_\theta$：吃 $x_{t_{n+1}}$（更噪的点），参与梯度更新
- **target network** $f_{\theta^-}$：吃 $\hat{x}^\phi_{t_n}$（经 ODE solver 去了一步的点），**不参与反传**
- $d(\cdot,\cdot)$ 用 LPIPS（实验最优）

**Step 4：EMA 更新 target network**

$$\theta^- \leftarrow \text{stopgrad}(\mu \theta^- + (1-\mu)\theta)$$

---

**CD 伪代码**

```python
def consistency_distillation(
    dataset, score_model_phi, N=18, mu=0.99,
    T=80, eps=0.002, rho=7
):
    # 构造时间步序列
    # t[i] = (eps^(1/rho) + i/(N-1) * (T^(1/rho) - eps^(1/rho)))^rho
    t = get_timesteps(eps, T, N, rho)  # shape: [N], t[0]=eps, t[N-1]=T

    # 初始化 online / target network（target 初始化为 online 的拷贝）
    theta = init_model()
    theta_minus = copy(theta)          # target network，不反传

    for step in range(max_steps):
        # ── 采样 ──────────────────────────────────────────────────
        x = sample_from_dataset(dataset)               # [B, C, H, W]
        n = randint(1, N-1, size=B)                    # 随机时间步索引
        t_next = t[n]                                  # t_{n+1}，较大噪声
        t_curr = t[n-1]                                # t_n，  较小噪声

        # 在 t_{n+1} 处加噪，得到 x_{t_{n+1}}
        noise = randn_like(x)
        x_next = x + t_next * noise                    # x_{t_{n+1}} ~ N(x, t_{n+1}^2 I)

        # ── 用 ODE solver 走一步，得到 x̂^phi_{t_n} ────────────────
        # Euler（简化示意，实际论文用 Heun 2阶）
        score = score_model_phi(x_next, t_next)        # s_phi(x_{t_{n+1}}, t_{n+1})
        x_curr_hat = x_next - (t_curr - t_next) * t_next * score
        # 注意：t_curr - t_next < 0（因为 t_curr < t_next），
        # 所以实际是向小 t 方向走

        # ── 计算 consistency loss ─────────────────────────────────
        # online network 吃更噪的点 x_{t_{n+1}}
        out_online = f_theta(x_next, t_next)           # [B, C, H, W]

        # target network 吃 ODE solver 走过一步的点 x̂^phi_{t_n}
        with torch.no_grad():
            out_target = f_theta_minus(x_curr_hat, t_curr)

        # d(·,·) = LPIPS（论文最优选择）
        loss = lambda_weight(t_curr) * LPIPS(out_online, out_target)
        loss = loss.mean()

        # ── 反传更新 online network ───────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── EMA 更新 target network（不参与梯度）────────────────────
        with torch.no_grad():
            for p, p_minus in zip(theta.parameters(), theta_minus.parameters()):
                p_minus.data = mu * p_minus.data + (1 - mu) * p.data
```

------

### 4.2Consistency Training (CT)

不依赖 $s_\phi$，直接用数据构造"伪相邻点对"：对同一个 $x$，分别加 $t_{n+1}$ 和 $t_n$ 的噪声，得到 $(x + t_{n+1}z, x + t_nz)$。这两个点在理想 PF ODE 下属于**同一条轨迹**（因为 $z$ 相同），所以也应满足 self-consistency。

**关键推导**

CT Loss 从 CD Loss 推来。用无偏 score 估计：

$$\nabla \log p_t(x_t) = -\mathbb{E}\left[\frac{x_t - x}{t^2} \bigg| x_t\right] \approx -\frac{x_t - x}{t^2}$$

代入 Euler ODE solver 的相邻点构造公式，化简得到：

$$\hat{x}^\phi_{t_n} \approx x + t_n z \quad \text{（当 } \Delta t \to 0\text{）}$$

因此 CD 中的 $(\hat{x}^\phi_{t_n}, x_{t_{n+1}})$ 变成了 $(x + t_n z,; x + t_{n+1} z)$，**完全从数据构造，无需 score model**。

**CT Loss**：

$$\mathcal{L}_{\text{CT}}^N(\theta, \theta^-) = \mathbb{E}\left[\lambda(t_n) \cdot d\left(f_\theta(x + t_{n+1}z; t_{n+1}); f_{\theta^-}(x + t_n z,; t_n)\right)\right]$$

**动态 N 调度的必要性**

CT loss 与真实 CD loss 之间的误差是 $O(\Delta t)$。$\Delta t$ 由 $N$ 决定：

- $N$ 小 → $\Delta t$ 大 → bias 大（CT loss 和 CD loss 差距大），但 variance 小，**收敛快**
- $N$ 大 → $\Delta t$ 小 → bias 小（接近真实 CD），但 variance 大，**收敛慢**

所以策略是：**训练初期用小 $N$ 快速收敛，后期逐渐增大 $N$ 提高精度**。$\mu$ 也随 $N$ 动态调整（$N$ 增大时 $\mu$ 相应增大，让 target network 更稳定）。

---

**CT 伪代码**

```python
def consistency_training(
    dataset, N_schedule, mu_schedule,
    T=80, eps=0.002, rho=7
):
    theta = init_model()
    theta_minus = copy(theta)
    k = 0  # 全局训练步数

    for step in range(max_steps):
        # ── 当前步的 N 和 mu（动态调度）────────────────────────────
        N   = N_schedule(k)       # 随训练进度增大，例如从 2 增到 150
        mu  = mu_schedule(k)      # 随 N 增大，例如 exp(s0 * log(mu0) / N)
        t   = get_timesteps(eps, T, N, rho)   # 重新离散化时间步

        # ── 采样 ──────────────────────────────────────────────────
        x = sample_from_dataset(dataset)      # [B, C, H, W]
        n = randint(1, N-1, size=B)           # 随机时间步索引

        t_next = t[n]                         # t_{n+1}
        t_curr = t[n-1]                       # t_n  （< t_{n+1}）

        z = randn_like(x)                     # 同一个噪声向量！

        # ── 构造伪相邻点对（无需 score model）──────────────────────
        # 关键：同一个 z，不同的噪声级别
        x_next = x + t_next * z              # x_{t_{n+1}} = x + t_{n+1} * z
        x_curr = x + t_curr * z              # x_{t_n}     = x + t_n     * z
        # 在理想 PF ODE 下，这两个点在同一条轨迹上

        # ── 计算 CT loss ───────────────────────────────────────────
        out_online = f_theta(x_next, t_next)  # online network 吃 t_{n+1} 的点

        with torch.no_grad():
            out_target = f_theta_minus(x_curr, t_curr)  # target network 吃 t_n 的点

        loss = lambda_weight(t_curr) * LPIPS(out_online, out_target)
        loss = loss.mean()

        # ── 反传更新 online network ───────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── EMA 更新 target network ───────────────────────────────
        with torch.no_grad():
            for p, p_minus in zip(theta.parameters(), theta_minus.parameters()):
                p_minus.data = mu * p_minus.data + (1 - mu) * p.data

        k += 1
```

------

**两种方法的核心差异对比**

```
CD：
  x ──加噪到 t_{n+1}──→ x_{t_{n+1}}
                              │
                        ODE solver (s_phi)
                              │
                              ↓
                         x̂^phi_{t_n}          ← 需要 score model
                              │
      f_theta(x_{t_{n+1}})  vs  f_{theta^-}(x̂^phi_{t_n})

CT：
  x ──同一个 z──→ x + t_{n+1}·z     x + t_n·z
                       │                  │
                  f_theta(·)         f_{theta^-}(·)   ← 不需要 score model
                  (online)           (target)
```

**CT 的代价**：由于用了近似的 score 估计，CT loss 和真实 CD loss 之间存在 $O(\Delta t)$ 的 bias，需要动态增大 $N$ 来逐步消除，导致训练更复杂、效果稍差于 CD（CIFAR-10 单步：CT FID 8.70 vs CD FID 3.55）。

---

## 5. 多步采样（Algorithm 1）

```
x ← f_θ(x̂_T, T)
for n = 1 to N-1:
    z ~ N(0, I)
    x̂_{τ_n} ← x + sqrt(τ_n² - ε²) · z    # 重新加噪
    x ← f_θ(x̂_{τ_n}, τ_n)                  # 再去噪
return x
```

时间点 ${\tau_1 > \tau_2 > \cdots > \tau_{N-1}}$ 用 ternary search 贪心选取（最小化 FID）。



