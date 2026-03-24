---
tags:
  - 流匹配	
  - 条件流匹配	
---

# Flow Matching 流匹配

> [!INFO] 文档信息
>
> 创建时间：2025-11-26 | 更新时间：2026-3-24
> 
> 原文链接**[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)** 



## ODE Solver 是什么

给定一个 ODE：
$$
\frac{dx}{dt} = v_t(x_t)
$$
已知初始状态 $x_0$，想求 $x_1 $。解析解几乎不存在（因为 $v_t $ 是一个神经网络），所以用数值方法逼近。

------

**最基础的：Euler Method**

把时间切成 $N $ 步，步长 $h = 1/N $：
$$
x_{t+h} = x_t + h \cdot v_t(x_t)
$$
这就是展开一阶导数逼近。每步用当前点的导数往前走一小步。

误差是 $O(h^2) $（局部），累积误差 $O(h) $。步数越多越准，但每步都要调用一次神经网络（一次 NFE）。

------

**为什么需要更高阶的方法**

Euler 的问题是轨迹弯曲时误差大。更高阶的方法通过在**一步内多次采样导数**来更好地估计曲率。

**Midpoint（RK2）：**
$$
k_1 = v_t(x_t)
$$
一步用 2 次 NFE，但误差降到 $O(h^3) $。

**RK4（经典四阶）：**
$$
x_{t+h} = x_t + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$
一步 4 次 NFE，误差 $O(h^5) $。论文里实验用的 Euler / Midpoint / RK4 就是这三个。

------

**Adaptive Solver（论文用的 dopri5）**

上面都是**固定步长**。论文评估 likelihood 用的 `dopri5` 是**自适应步长**的：

- 同时用 4 阶和 5 阶估计，比较差异来估计误差
- 误差大的地方自动缩小步长，误差小的地方放大步长
- 可以设置 tolerance（`atol=rtol=1e-5`），保证精度

代价是 NFE 不固定，轨迹越弯需要越多步。这也是为什么 Score Matching 训练的模型 NFE 会高达 400+，而 FM-OT 只需要 100 出头——OT 轨迹更直，曲率更小，solver 可以用大步长。



好，现在有了 ODE 的基础，可以把出发点讲得更清楚。

------

## 用 ODE 做生成模型

目标很简单：训练一个神经网络 $v_\theta(x, t)$，使得解这个 ODE：

$$\frac{dx}{dt} = v_\theta(x_t, t), \quad x_0 \sim \mathcal{N}(0, I)$$

从 $t=0$ 积分到 $t=1$ 之后，$x_1 $ 的分布就是数据分布 $q(x)$。

这个想法就是 CNF。问题是**怎么训练这个 $v_\theta$？**

------

**之前的训练方法为什么不行？**

**方法一：最大似然**

直接优化 $\log p_1(x_1)$，需要在训练时解 ODE 做反向传播，极其昂贵，高维图像完全 scale 不了。

**方法二：借道 Diffusion / Score Matching**

Song et al. 发现 diffusion SDE 对应一个概率流 ODE，可以用 score matching 来训练。但这把你**锁死在 diffusion 定义的特定路径上**——VP、VE 这些，路径弯曲，采样需要大量步数。

**方法三：其他 simulation-free 方法**

要么有 intractable 的积分，要么梯度有偏。

------

## 指定概率路径

既然问题是"找一个 $v_t $ 使概率从噪声流向数据"，为什么不**直接构造这个 $v_t $**？

关键观察：不需要通过 diffusion 绕一圈。可以直接问：

> 对于每一个数据点 $x_1 $，我能不能构造一条从噪声 $x_0$ 到 $x_1 $ 的路径，然后把这些路径的 vector field 叠加起来？

这就是 **conditional 构造**的思路——每个数据点单独定义一条路径，marginal 的 vector field 自然由所有 conditional 叠加得到。

然后再加上 **CFM 的等价性**（Theorem 2）：回归 conditional vector field 和回归 marginal vector field 梯度相同。

于是训练变成了：

$$\mathcal{L}_{CFM} = \mathbb{E}_{t, x_1, x_0} \left\| v_\theta(\psi_t(x_0), t) - \frac{d}{dt}\psi_t(x_0) \right\|^2$$

每次采样一个 $x_1$、一个 $x_0$、一个 $t$，构造插值点，回归那个点上的方向。**完全不需要解 ODE**。

------

**OT 路径**

路径可以任意选，Lipman 选了最自然的一个——线性插值：

$$\psi_t(x_0) = (1-t)x_0 + t x_1$$

这在数学上对应两个高斯之间的 Wasserstein-2 最优传输，物理直觉就是：**粒子走直线，匀速从 $x_0$ 运动到 $x_1$**。

训练目标因此变成极其简单的常数方向 $x_1 - x_0$，不随时间变化，网络更容易拟合，采样时轨迹也更直，solver 需要的步数更少。

> [!note]
>
> **与其通过 diffusion 间接定义路径再用 score matching 训练，不如直接指定路径（比如直线），然后用简单的回归来训练 vector field。**

---

## 概率速度场推导

> [!note]
>
> 我只知道每个数据点 $x_1 $ 对应的 conditional vector field $u_t(x|x_1)$，怎么能得到整个分布的 marginal vector field $u_t(x)$？

换句话说，单个粒子的速度，怎么推出整个概率密度的速度场？

------

**先理解 Marginal 的构造**

marginal probability path 就是把所有 conditional path 按数据分布加权平均：

$$p_t(x) = \int p_t(x|x_1)q(x_1)dx_1$$

直觉上很自然：在时刻 $t$，整个分布就是"所有数据点各自带着一团高斯云，叠加在一起"。

------

**关键问题：速度场怎么叠加？**

你可能会直觉地想：速度场也直接加权平均？

$$u_t(x) \stackrel{?}{=} \int u_t(x|x_1)q(x_1)dx_1$$

**这是错的。** 直接对 vector field 加权平均不能保证生成正确的 marginal path。

正确的叠加方式是**按照 conditional 概率加权**：

$$u_t(x) = \int u_t(x|x_1) \frac{p_t(x|x_1)q(x_1)}{p_t(x)}dx_1$$

注意权重是 $\frac{p_t(x|x_1) q(x_1)}{p_t(x)}$，也就是**给定位置 $x$ 在时刻 $t$，这个 $x$ 来自数据点 $x_1 $ 的后验概率**。

------

**为什么是后验加权？直觉解释**

想象时刻 $t$，空间中某个位置 $x$。

这个位置上有很多"粒子流"经过，每个数据点 $x_1 $ 都在这里贡献一些粒子，贡献量正比于 $p_t(x|x_1) q(x_1)$。

每股粒子流的速度是 $u_t(x|x_1)$。

那么这个位置上粒子的**平均速度**，自然是按粒子数量加权的平均：

$$u_t(x) = \frac{\int u_t(x|x_1)\cdot p_t(x|x_1) q(x_1) dx_1}{\int p_t(x|x_1) q(x_1) dx_1} = \int u_t(x|x_1) \frac{p_t(x|x_1) q(x_1)}{p_t(x)} dx_1$$

分母就是 $p_t(x)$，所以权重归一化了。

------

### Theorem 1

需要验证：用这个 $u_t(x)$ 和 $p_t(x)$ 满足**连续性方程**：

$$\frac{\partial}{\partial t} p_t(x) + \text{div}(p_t(x), u_t(x)) = 0$$

连续性方程是充要条件——满足它就说明 $u_t$ 确实生成了 $p_t$。

证明分两步：

**第一步：** 对 marginal 求时间导数

$$\frac{d}{dt}p_t(x) = \int \frac{d}{dt}p_t(x|x_1)q(x_1) dx_1$$

因为每个 conditional path 满足自己的连续性方程：

$$\frac{d}{dt}p_t(x|x_1) = -\text{div}\big(u_t(x|x_1)p_t(x|x_1)\big)$$

代入得：

$$\frac{d}{dt}p_t(x) = -\int \text{div}\big(u_t(x|x_1)p_t(x|x_1)\big)q(x_1)dx_1$$

**第二步：** 把积分和 div 交换（Leibniz rule）：

$$= -\text{div}\int u_t(x|x_1)p_t(x|x_1)q(x_1)dx_1$$

而右边正好是：

$$= -\text{div}\big(u_t(x)p_t(x)\big)$$

这就是 marginal 的连续性方程，得证。

------

> [!note]
>
> 回到刚才的问题：数据点的速度怎么能推出整个分布的速度场？

本质上不是"推出"，而是**构造**：

每个数据点 $x_1 $ 定义一条 conditional path 和对应的 conditional vector field。把所有这些 conditional vector field 按后验概率在每个位置 $x$ 加权叠加，得到的 marginal vector field 在数学上**保证满足连续性方程**，因此保证能把 $p_0$ 变换到 $p_1$。

关键是连续性方程的线性性——它对 $p_t$ 和 $p_t u_t$ 都是线性的，所以 conditional 的叠加自然传递到 marginal。

------

### 连续性方程和"速度场生成概率路径"等价

**先建立直觉**

想象概率密度 $p_t(x)$ 是一片沙子撒在空间里，每粒沙子都在速度场 $v_t(x)$ 的驱动下运动。

"速度场生成概率路径"的意思就是：沙子按照 $v_t $ 运动，整体的密度分布随时间的变化恰好是 $p_t$。

连续性方程就是这个物理过程的数学表达。

------

### 连续性方程的推导

考虑空间中一个小区域 $\Omega $。区域内的概率质量是 $\int_\Omega p_t(x) dx $。

这个质量的变化率 = 流入量 - 流出量：

$$\frac{d}{dt}\int_\Omega p_t(x)\, dx = -\oint_{\partial\Omega} p_t(x)\, v_t(x) \cdot \hat{n}\, dS$$

右边用散度定理变成体积分：

$$\frac{d}{dt}\int_\Omega p_t(x)\, dx = -\int_\Omega \text{div}(p_t(x)\, v_t(x))\, dx$$

因为 $\Omega$ 是任意区域，被积函数必须处处相等：

$$\frac{\partial p_t(x)}{\partial t} + \text{div}(p_t(x)\, v_t(x)) = 0$$

这就是连续性方程。它说的就是**概率质量守恒**——没有概率凭空产生或消失，只是被速度场搬运。

------

**等价性**

所以"$v_t $ 生成 $p_t $"和"$v_t $、$p_t $ 满足连续性方程"是同一件事的两种说法：

- 生成的意思：从 $p_0 $ 出发，按 $v_t $ 运动，时刻 $t $ 的密度恰好是 $p_t $
- 连续性方程：密度随时间的变化完全由速度场的散度决定

满足连续性方程是充要条件。这也是为什么 Theorem 1 的证明只需要验证连续性方程就够了。

------

## CFM 和 FM 梯度等价

先写出两个 loss

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t,\, p_t(x)}\| v_\theta(x,t) - u_t(x) \|^2$$

FM 回归 marginal vector field（intractable），CFM 回归 conditional vector field（tractable）。

------

**展开范数**

对任意向量 $a, b, c $，有：
$$
\|a - b\|^2 = \|a - c\|^2 + 2\langle a-c,\, c-b \rangle + \|c - b\|^2
$$
把 $a = v_\theta $，$b = u_t(x|x_1) $，$c = u_t(x) $ 代入 CFM loss：
$$
\mathcal{L}_{CFM} = \mathbb{E}\Big[\| v_\theta - u_t(x) \|^2 + 2\langle v_\theta - u_t(x),\, u_t(x) - u_t(x|x_1)\rangle + \| u_t(x) - u_t(x|x_1) \|^2\Big]
$$
第一项就是 $\mathcal{L}_{FM} $。第三项不含 $\theta $，是常数。

**关键：第二项的期望为零。**

------

**为什么交叉项消失**

第二项的期望：

$$\mathbb{E}_{t,\, q(x_1),\, p_t(x|x_1)}\Big[\langle v_\theta(x,t) - u_t(x),\; u_t(x) - u_t(x|x_1)\rangle\Big]$$

先对 $x_1 $ 积分（固定 $x $ 和 $t $）：

$$\mathbb{E}_{q(x_1),\, p_t(x|x_1)}\Big[u_t(x) - u_t(x|x_1)\Big]$$

$$= \int \big(u_t(x) - u_t(x|x_1)\big)\, p_t(x|x_1)\, q(x_1)\, dx_1 $$

$$= u_t(x) \underbrace{\int p_t(x|x_1) q(x_1) dx_1}_{= p_t(x)} - \underbrace{\int u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1}_{= u_t(x) p_t(x)} $$

$$= u_t(x)\, p_t(x) - u_t(x)\, p_t(x) = 0 $$

第二个等号用的正是 marginal vector field 的定义式（上一节讲的后验加权）。

------

### 结论

$$\mathcal{L}_{CFM}(\theta) = \mathcal{L}_{FM}(\theta) + \text{const}$$

对 $\theta $ 求梯度，常数项消失：

$$\nabla_\theta \mathcal{L}_{CFM} = \nabla_\theta \mathcal{L}_{FM}$$

所以优化 CFM 和优化 FM **完全等价**。

------

**为什么这很重要**

FM loss 里的 $u_t(x)$ 需要对所有 $x_1 $ 积分，没有 closed form，根本没法算。

CFM loss 里的 $u_t(x|x_1) $ 是单个数据点的 conditional vector field，有 closed form（Theorem 3 给出了解析表达式）。

所以训练时每步只需要：采样 $t $，采样 $x_1 $，采样 $x_0 $，构造插值点 $\psi_t(x_0) $，计算 $u_t(\psi_t(x_0)|x_1) $，然后回归。完全 tractable，完全 simulation-free。

这在数学结构上和 denoising score matching 的 trick 完全一样——把 intractable 的 marginal 目标替换成 tractable 的 conditional 目标，两者梯度等价。
