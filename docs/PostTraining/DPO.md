---
tags:
  - 后训练
  - DPO


---

# DPO

> [!INFO] 文档信息
>
> 创建时间：2026-3-13 | 更新时间：2026-3-13
>
> 论文链接 [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
>



## 核心思路

在 DPO 出现之前，学术界普遍认为要对齐人类偏好，必须经过以下流程：

1. **训练奖励模型 (Reward Model):** 给几万对 `(好回答, 坏回答)`，让一个模型学会打分。
2. **强化学习 (PPO):** 用这个打分模型作为“老师”，通过 PPO 算法不断调整大模型的参数。

**痛点：** PPO 极其难调，不仅吃显存，而且对超参数非常敏感，稍有不慎模型就会崩溃。



于是作者提出了一个大胆的猜想：**一个训练良好的大模型，其输出概率本身就隐藏了“奖励”信息。**

如果模型对回答 A 的概率远高于回答 B，那本质上就意味着模型认为 A 的“奖励值”更高。既然如此，为什么我们还要费劲去训练一个额外的奖励模型呢？



作者通过数学公式证明了**最优奖励函数（Optimal Reward Function）可以完全由模型输出概率的对数比来表达。**

简单来说，他们推导出了一个恒等式，将“奖励值”替换成了“模型概率”。这样一来：

- **过去：** 最小化（模型预测分数与人类打分的差异）。
- **现在：** 最大化（好答案相对于坏答案的胜出概率）。

> 这一步直接把**强化学习**降维打击成了**二分类交叉熵损失（Binary Cross Entropy Loss）**，就像我们训练图片分类器一样简单。



DPO 的损失函数公式如下：

$$L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

这里有三个关键角色：

1. $\pi_\theta$：你正在训练的**新模型**。
2. $\pi_{ref}$：冻结住的**原始参考模型**（防止新模型跑偏）。
3. $\beta$：一个调节杠杆。$\beta$ 越大，模型越在意偏好数据；$\beta$ 越小，模型越倾向于保持原样。

**它的逻辑是：** 如果 $y_w$ 是好答案，那么新模型相对于旧模型的“进步程度”应该远大于坏答案 $y_l$ 的“进步程度”。

------

**论文的结论与贡献**

- **去掉奖励模型：** 节省了大量的计算资源和内存。
- **性能卓越：** 实验证明，DPO 在摘要生成和对话任务上，表现甚至优于复杂的 PPO。
- **稳定性：** DPO 是确定的优化过程，不会像 PPO 那样因为采样随机性而产生剧烈波动。

------



## 数学推导

**第一步：定义原始的 RL 优化目标**

在标准的 RLHF 中，我们的目标是找到一个策略（模型）$\pi_\theta$，使其在满足“不偏离原始模型 $\pi_{ref}$”的约束下，最大化奖励 $r(x, y)$。

数学表达式为：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} [r(x, y)] - \beta \mathbb{D}_{KL}(\pi(\cdot|x) \| \pi_{ref}(\cdot|x))$$

**$x$ 就是 Prompt（提示词/问题）**，来自于我们的训练数据集 $\mathcal{D}$。

**$y$ 就是 Response（回答/输出）**，是由当前正在训练的策略模型 $\pi$ 生成的。

**KL 散度（$\mathbb{D}_{KL}$）是一个正则项（惩罚项）。** 如果没有这个 KL 正则项，模型为了单纯追求奖励 $r(x, y)$ 的最大化，会产生“奖励破解（Reward Hacking）”。比如，它可能会发现只要输出“啊啊啊啊”就能骗过奖励模型拿到高分，从而彻底丧失正常的语言能力。KL 正则项的作用就是把模型“拉住”，强迫它**在尽量不偏离预训练模型（或 SFT 模型） $\pi_{ref}$ 原有概率分布的前提下**，去争取更高的奖励。

------

**第二步：解出最优策略的解析解**

为了求出最优的策略 $\pi^*(y|x)$，我们需要解上面那个带有 KL 散度的优化问题。为了书写方便，我们针对一个固定的输入 $x$，省略掉外层的期望，把 KL 散度展开：

$$\max_{\pi} \sum_y \pi(y|x) \left[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right]$$

这是一个典型的带约束优化问题，因为所有的概率加起来必须等于 1，即约束条件为：$\sum_y \pi(y|x) = 1$。

我们可以使用**拉格朗日乘数法（Lagrange Multipliers）**来求解。构造拉格朗日函数 $L$（引入乘子 $\lambda$）：

$$L(\pi, \lambda) = \sum_y \pi(y|x) \left[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right] + \lambda \left( 1 - \sum_y \pi(y|x) \right)$$

接下来，我们对 $\pi(y|x)$ 求偏导，并令其等于 0，寻找极值点：

$$\frac{\partial L}{\partial \pi(y|x)} = r(x,y) - \beta \left( \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + 1 \right) - \lambda = 0$$

通过简单的移项和代数变形，我们可以解出 $\pi(y|x)$：

$$\log \frac{\pi(y|x)}{\pi_{ref}(y|x)} = \frac{1}{\beta} r(x,y) - 1 - \frac{\lambda}{\beta}$$

两边同时取指数（$\exp$）：

$$\pi(y|x) = \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right) \exp\left(-1 - \frac{\lambda}{\beta}\right)$$

因为概率的总和必须为 1（$\sum_y \pi(y|x) = 1$），所以后面那一坨与 $y$ 无关的常数 $\exp(-1 - \frac{\lambda}{\beta})$ 实际上就扮演了**归一化常数**的倒数。我们把它记为 $\frac{1}{Z(x)}$。



于是，我们就得到了最优策略的闭式解：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp \left( \frac{1}{\beta} r(x, y) \right)$$

**关键转折点来了：**

如果我们把上面的公式反过来写，用 $\pi_r$ 和 $\pi_{ref}$ 来表示奖励函数 $r(x, y)$，会得到：

$$r(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

> **这就是 DPO 的核心直觉：** 奖励函数可以用“当前模型”与“参考模型”的概率对数比来完美表达。

------

**第三步：代入 Bradley-Terry 偏好模型**

现在我们有了奖励的表达式，但我们依然不知道具体的奖励值是多少。这时，论文引入了处理人类偏好的经典模型——**Bradley-Terry (BT) 模型**。

BT 模型认为，人类在两个回答 $(y_w, y_l)$ 中选择 $y_w$ 的概率取决于它们的奖励差：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

*(注：$\sigma$ 是 Sigmoid 函数)*



根据上面的推导，归一化常数（也叫配分函数 Partition Function）$Z(x)$ 的数学定义是：

$$Z(x) = \sum_y \pi_{ref}(y|x) \exp \left( \frac{1}{\beta} r(x, y) \right)$$

**它的物理意义是：** 对于给定的问题 $x$，模型**所有可能的输出序列 $y$** 的某种指数奖励的加权总和。

**为什么它是个大麻烦，必须被消去？**

在语言模型中，$y$ 是一整个句子。假设词表大小是 50,000，生成一个长度为 100 个 Token 的句子，那么“所有可能的输出序列”的数量是 $50000^{100}$。

这是一个天文数字！我们根本**不可能**穷举所有的句子去计算它们的奖励 $r(x,y)$ 然后求和。因此，$Z(x)$ 在计算上是不可解（Intractable）的。传统的强化学习（如 PPO）通过极其复杂的采样和动态规划来绕过这个问题。

而 DPO 的伟大之处，就是通过 Bradley-Terry 模型的减法机制，让这个不可计算的 $Z(x)$ 像魔法一样自己抵消掉了：

$$r(x, y_w) - r(x, y_l) = \left( \beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} + \beta \log Z(x) \right)$$

$Z(x)$ 被减没后，我们就彻底摆脱了这个计算上的黑洞，可以直接用现成的模型概率来进行反向传播了。

最终剩下的部分就是 DPO 的损失函数项：

$$P(y_w \succ y_l | x) = \sigma \left( \beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

------

**总结**

1. **避开了 $Z(x)$：** 在传统的强化学习里，计算配分函数 $Z(x)$ 需要对整个输出空间求和，这在 LLM 里是不可能的。DPO 通过减法巧妙地消去了它。
2. **无需奖励模型：** 我们不再需要显式地训练一个 $r(x, y)$，而是直接让策略模型 $\pi$ 去拟合人类的偏好概率。
3. **确定性优化：** 现在的损失函数变成了负对数似然（Negative Log-Likelihood），本质上就是一个二分类任务。

**这就解释了为什么 DPO 比 PPO 稳定得多：** 它把一个复杂的动态博弈过程（策略更新带动奖励更新，奖励更新反过来再指导策略）变成了一个静态的、目标明确的单步优化。