# 强化学习学习路径：从RL基础到GRPO

------

## 推荐路径

### Phase 1：最小必要 RL 知识（3-5天）

不需要读教材，只需要掌握以下概念框架：

**核心概念清单**

| 概念                          | 需要掌握的程度        |
| ----------------------------- | --------------------- |
| MDP $(S, A, R, \gamma)$       | 知道定义即可          |
| Policy $\pi_\theta(a|s)$      | 理解为参数化分布      |
| Value Function $V^\pi, Q^\pi$ | 理解 Bellman 方程结构 |
| Advantage $A^\pi(s,a)$        | **重点**，GRPO 的核心 |
| Monte Carlo vs TD             | 知道 trade-off        |

**推荐直接读**：Spinning Up in Deep RL（OpenAI，网页文档，非论文）+ 李宏毅 PPO 讲座视频，**不要读 Sutton 教材**，性价比极低。

------

### Phase 2：PPO 精读（必读论文）

**Schulman et al., 2017** — *Proximal Policy Optimization Algorithms*

核心是两件事：

**1. CLIP 目标**

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ 是重要性采样比。

**2. GAE（Generalized Advantage Estimation）**

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\lambda$ 控制 bias-variance tradeoff。

**读论文还是看框架**：PPO **必须读论文**，它只有8页，公式清晰，是后续所有方法的参照基线。

------

### Phase 3：RLHF 范式（理解应用背景）

理解 InstructGPT / RLHF pipeline：

```
SFT Model → Reward Model (RM) 训练 → PPO 优化 policy
```

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} [r_\phi(x, y)] - \beta \cdot \mathbb{D}_{\text{KL}}[\pi_\theta | \pi_{\text{ref}}]$$

KL 散度项防止 reward hacking，$\pi_{\text{ref}}$ 是 SFT 基准模型。

**推荐读**：InstructGPT 论文（Ouyang et al., 2022），重点看 Section 3 的训练细节。

------

### Phase 4：GRPO 精读（核心目标）

**DeepSeekMath / DeepSeek-R1** 引入，**Group Relative Policy Optimization**

**动机**：PPO 需要 Critic（Value Network），对 LLM 来说成本极高（需要额外一个同等规模模型）。GRPO 的核心思想是**用组内相对奖励替代 Critic 估计的 Baseline**。

**训练目标**：

$$\mathcal{L}^{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(r_{i}(\theta)\hat{A}*i,\ \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}\*i\right) - \beta \mathbb{D}\*{\text{KL}}[\pi*\theta | \pi_{\text{ref}}]\right]$$

Advantage 的计算方式：

$$\hat{A}_i = \frac{r_i - \text{mean}({r_1, \dots, r_G})}{\text{std}({r_1, \dots, r_G})}$$

对同一个 prompt $q$，采样 $G$ 个输出 ${o_1, \dots, o_G}$，用组内均值归一化作为 baseline，**完全去掉 Value Network**。

**与 PPO 对比**：

|                | PPO                              | GRPO                  |
| -------------- | -------------------------------- | --------------------- |
| Baseline 来源  | Critic Network（参数量 = Actor） | 组内 reward 统计量    |
| 显存开销       | ~2x Actor                        | ~1x Actor             |
| Advantage 估计 | GAE（时序 TD）                   | 跨样本归一化          |
| 适用场景       | 连续控制、游戏                   | LLM 推理、生成任务    |
| 必要条件       | 密集 reward                      | 稀疏/结果 reward 即可 |

**必读论文**：

- DeepSeekMath（2024）—— GRPO 原始出处
- DeepSeek-R1（2025）—— GRPO + 长链推理的完整实践

------

## 这个方向在研究什么

### 当前热点研究问题

**1. Reward 设计**

- Rule-based reward（格式奖励 + 正确性奖励）vs. RM-based reward
- Reward hacking / reward gaming 的检测与缓解
- Process Reward Model（PRM，step-level reward）vs. ORM

**2. 训练稳定性**

- KL 散度崩溃（entropy collapse）问题
- Clip 范围 $\epsilon$ 的调度策略
- 长序列下的梯度方差问题

**3. 去掉 KL 约束的变体**

- DAPO（Qwen team）：移除 token-level KL，改为 clip-higher 策略
- Dr. GRPO：分析 GRPO 中的统计偏差，提出去偏估计

**4. GRPO 的泛化应用（你最可能做的方向）**

| 应用域             | 做法                               | 代表工作                     |
| ------------------ | ---------------------------------- | ---------------------------- |
| 数学推理           | rule-based correctness reward      | DeepSeek-R1, STILL-3         |
| 代码生成           | 编译/测试通过率作为 reward         | CodeR1                       |
| 视觉语言模型       | 多模态 reward（bbox 精度等）       | R1-V, VL-Thinking            |
| 医疗/科学          | 领域规则 reward                    | Med-R1                       |
| **Diffusion 对齐** | reward 作用于 denoising trajectory | DDPO, DPOK（与你背景最相关） |

------

## 给你的具体建议

**最优学习顺序**：

```
Spinning Up (概念速通, 2天)
    → PPO 论文精读 (1天)
    → InstructGPT 论文 Section 3 (半天)
    → DeepSeekMath 论文 (GRPO 原始推导, 1天)
    → DeepSeek-R1 技术报告 (工程实践, 1天)
    → 选一个具体应用方向的论文
```

**需要读论文还是看代码**：

- PPO / GRPO 的**数学推导必须读论文**，否则调参没有直觉
- 工程实现推荐看 **TRL (HuggingFace)** 或 **OpenRLHF** 的源码，比读代码教程高效得多
- GRPO 的 PyTorch 核心逻辑不超过 100 行，建议自己从头实现一遍 bandit 版本

**与你 Diffusion 背景的连接点**：

DDPO（Denoising Diffusion Policy Optimization）把 denoising 过程建模为 MDP，直接用 PPO 优化 reward（如 CLIP score、人类偏好），这是你背景与 RL 的天然交叉点，可以作为入门实践项目。