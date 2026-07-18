### 常见深度学习优化器概览

#### 1. SGD 系列

**Vanilla SGD**
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
$$
**SGD + Momentum**
$$
v_{t+1} = \beta v_t + \nabla_\theta \mathcal{L}, \quad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$
**Nesterov Accelerated Gradient (NAG)**
$$
v_{t+1} = \beta v_t + \nabla_\theta \mathcal{L}(\theta_t - \eta \beta v_t), \quad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$
前瞻梯度使收敛更稳定，特别是在凸目标上有理论保证。

------

#### 2. 自适应学习率系列

**AdaGrad**
$$
G_t = G_{t-1} + g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t
$$
问题：$G_t $ 单调递增，后期学习率趋近于零。

**RMSProp**（修复 AdaGrad 的指数移动平均版本）
$$
v_t = \beta v_{t-1} + (1-\beta) g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} g_t
$$
**Adam**（目前最主流）
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一阶矩)}
$$
默认参数：$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8} $

------

#### 3. Adam 变体

| 变体        | 核心改动                                                     | 适用场景                             |
| ----------- | ------------------------------------------------------------ | ------------------------------------ |
| **AdamW**   | weight decay 从梯度项解耦：$\theta \leftarrow \theta - \lambda\theta $（独立于自适应缩放） | Transformer 标配，DiT/SiT 训练首选   |
| **AdaMax**  | 二阶矩改用 $\ell_\infty $ 范数替代 $\ell_2 $                 | 理论更稳定，实践差异不大             |
| **AMSGrad** | $\hat{v}_t = \max(\hat{v}_{t-1}, v_t) $，保证 $v_t $ 单调不减 | 修复 Adam 不收敛反例，但实践提升有限 |
| **Adan**    | 引入 Nesterov + 三阶矩（梯度差分）                           | 部分视觉任务加速收敛                 |
| **Lion**    | 仅用梯度符号更新，内存节省                                   | 大模型，对 lr 敏感                   |
| **Sophia**  | 用 Hessian 对角近似做二阶缩放                                | LLM 预训练，收敛步数↓                |

------

#### 4. 关于 Adam vs AdamW 的关键区别

标准 Adam 的 weight decay 实现：
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}(\hat{m}_t + \lambda\theta_t)
$$
这导致 weight decay 被自适应缩放吃掉（参数更新幅度大的维度 decay 反而弱）。

AdamW 的正确实现：
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t - \eta\lambda\theta_t
$$
decay 项独立于梯度缩放，正则化效果更干净。**训练 DiT/SiT 用 AdamW 而非 Adam。**

------

#### 5. 选择建议

```
Transformer/Diffusion 训练   → AdamW (lr=1e-4, wd=1e-2)
快速实验/小模型              → Adam
需要泛化性能压榨              → AdamW + cosine LR schedule
分布式大模型                  → Lion / Sophia（显存友好）
凸优化/理论研究               → SGD + Momentum + LR schedule
```

------

#### 6. 二阶方法（补充）

**K-FAC**：用 Kronecker 分解近似 Fisher 信息矩阵，理论上收敛步数最少，但每步计算开销大，工程实现复杂，少数大规模实验室使用。

### 自适应学习率：核心思想

#### 问题动机

SGD 对所有参数用同一个 $\eta $。但实际上：

- 稀疏特征（如 embedding）：梯度偶尔很大，需要小 lr
- 密集特征：梯度稳定，可以用大 lr

**自适应的目标**：让每个参数维度有自己的有效学习率。

------

#### AdaGrad：累积历史梯度平方

$$
G_t^{(i)} = \sum_{\tau=1}^{t} g_\tau^{(i)2}
$$

**直觉**：某维度历史梯度大 → $G_t $ 大 → 有效 lr 小；反之亦然。

**问题**：$G_t $ 只增不减，训练后期所有维度学习率都趋近于 0，训练停滞。

------

#### RMSProp：用指数移动平均替代累加

把"所有历史"改成"近期历史"：
$$
v_t^{(i)} = \beta v_{t-1}^{(i)} + (1-\beta) g_t^{(i)2}
$$
$v_t $ 是梯度平方的 EMA，会遗忘远古历史，学习率不会归零。

------

#### Adam：动量 + 自适应学习率

同时维护一阶矩（动量）和二阶矩（自适应缩放）：
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \leftarrow \text{梯度的 EMA（方向）}
$$
**为什么需要 Bias Correction？**

初始 $m_0 = v_0 = 0 $，早期估计严重偏低。例如 $t=1 $：
$$
m_1 = (1-\beta_1)g_1 \approx 0.1 g_1 \quad \text{（实际梯度被压缩了10倍）}
$$
修正：
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
随着 $t $ 增大，$1-\beta^t \to 1 $，修正项自动消失。

最终更新：
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$
**有效学习率**的实际大小约为：
$$
\eta_{\text{eff}} \approx \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot |\hat{m}_t|
$$
当梯度稳定时 $\hat{m}_t \approx \sqrt{\hat{v}_t} $，有效 lr $\approx \eta $；梯度不稳定时自动缩小。

------

#### 三者对比

|         | 缩放依据         | 遗忘机制         | 动量             |
| ------- | ---------------- | ---------------- | ---------------- |
| AdaGrad | 全历史梯度平方和 | 无               | 无               |
| RMSProp | 近期梯度平方 EMA | 有（$\beta $）   | 无               |
| Adam    | 近期梯度平方 EMA | 有（$\beta_2 $） | 有（$\beta_1 $） |

Adam = RMSProp + Momentum + Bias Correction，是目前的事实标准。