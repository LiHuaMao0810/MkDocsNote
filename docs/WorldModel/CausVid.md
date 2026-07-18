# CausVid: From Slow Bidirectional to Fast Causal Video Generators

**arXiv: 2412.07772 | CVPR 2025 | MIT + Adobe**

------

## TL;DR

CausVid 将预训练的**双向注意力 (bidirectional)** 视频扩散 Transformer 蒸馏为**因果自回归 (causal autoregressive)** 的少步生成器，同时解决了两个正交问题：

1. **架构问题**：bidirectional → causal，使得帧间流式生成成为可能
2. **速度问题**：50步 → 4步，通过 DMD distillation 实现

核心结果：初始延迟仅 1.3 秒，之后以约 9.4 FPS 的速率流式输出帧，支持 streaming video-to-video translation、image-to-video 和 dynamic prompting 的零样本迁移。

------

## 问题建模

**Bidirectional 模型的根本瓶颈：**

设视频为 $\mathbf{x} = {x^1, x^2, \ldots, x^T}$，在 DiT 架构中，全帧双向注意力使得：

$$ p_\theta(x^t_\tau | \mathbf{x}_{\neq \tau}^t) $$

每一帧的去噪都依赖整个序列（包括未来帧），导致：

- 无法流式输出
- 显存/计算随帧数 $T$ 二次增长
- 典型案例：生成 128 帧视频需要 219 秒

**Autoregressive 模型的问题**：error accumulation，每帧基于已生成（可能存在误差的）历史帧预测，误差随时间累积，长视频质量崩溃。

---

## CausVid Section 4 详细解析

论文的完整训练流程分两个阶段（对应 Fig. 5）：**Student Initialization（ODE 回归预热）→ Asymmetric Distillation with DMD**。

### 4.1 Causal Architecture

**分块因果注意力机制**

模型在 3D VAE 的 latent space 操作。VAE 把每 16 帧视频压缩为 5 帧 latent chunk，模型按 chunk 为单位做扩散。

注意力 mask 的设计是整篇论文的架构核心：

$$ M_{i,j} = \begin{cases} 1, & \text{if } \left\lfloor \frac{j}{k} \right\rfloor \leq \left\lfloor \frac{i}{k} \right\rfloor \ 0, & \text{otherwise} \end{cases} $$

其中 $i, j$ 是帧索引，$k$ 是 chunk size（=5 latent frames）。

**这个 mask 的语义**：

- **chunk 内部**：$\lfloor j/k \rfloor = \lfloor i/k \rfloor$，同一 chunk 内全 attend，是 bidirectional 的（局部一致性）
- **chunk 之间**：$\lfloor j/k \rfloor < \lfloor i/k \rfloor$，当前 chunk 只能 attend 历史 chunk，不能看未来，是 causal 的

用图示来理解，假设 3 个 chunk（A B C），attention 矩阵是 block lower-triangular：

```
      Chunk A   Chunk B   Chunk C
Chunk A  [full]    [0]       [0]
Chunk B  [full]   [full]     [0]
Chunk C  [full]   [full]    [full]
```

这与 GPT-style 的 token 级 causal mask 的区别在于：**粒度是 chunk 而非单帧**，原因是 3D VAE decoder 最少需要一个 latent chunk 才能解码出像素，所以 chunk 是最小的"可输出单元"，逐帧的细粒度 causality 没有实际意义。

**训练时**借用 teacher 的 bidirectional 预训练权重初始化，只改 attention mask，收敛更快。

------

### 4.2 Bidirectional → Causal 非对称蒸馏

**为什么不用 Causal Teacher 直接蒸馏？**

论文明确否定了"先把 bidirectional teacher 改成 causal，再蒸馏"的 naive 做法：

> "causal diffusion models typically underperform their bidirectional counterparts, training a student model from a weaker causal teacher inherently limits the student's capabilities. Moreover, issues such as error accumulation would propagate from teacher to student."

即 causal teacher 本身质量受限，且 error accumulation 会直接传给 student。

**非对称蒸馏的做法**

**Teacher（bidirectional，冻结）**：作为 $s_\text{data}$，即真实数据分布的 score estimator
 **Student（causal，训练）**：作为生成器 $G_\phi$
 **Fake score（online 更新）**：$s_\text{gen}$，估计 student 输出分布的 score

DMD 损失梯度（来自 Eq. 4）：
$$
\nabla_\phi \mathcal{L}_\text{DMD} \approx -\mathbb{E}_t \int \left( s_\text{data}\bigl(\Psi(G_\phi(\epsilon), t),\, t\bigr) - s_\text{gen}\bigl(\Psi(G_\phi(\epsilon), t),\, t\bigr) \right) \frac{dG_\phi(\epsilon)}{d\phi}\, d\epsilon
$$
直觉：$s_\text{data} - s_\text{gen}$ 是"真实分布的 score"减"当前生成分布的 score"，其方向就是把生成分布推向真实分布的梯度信号。

**Algorithm 1 的完整流程**

```python
# 输入：少步去噪时间步集合 T, 视频长度 N, chunk size k,
#       预训练 bidirectional teacher s_data, 数据集 D

# 初始化
student G_phi ← ODE regression 预热（Sec 4.3）
s_gen ← s_data  # 用 teacher 权重初始化 fake score

while training:
    # ---- 数据准备 ----
    {x^i_0}^L_{i=1} ~ D          # 采一段真实视频，分成 L 个 chunk
    {t^i}^L_{i=1} ~ Uniform(T)   # 每个 chunk 独立采样噪声水平
    x^i_{t^i} = α_{t^i} x^i_0 + σ_{t^i} ε^i  # 各 chunk 独立加噪

    # ---- Student 前向 ----
    x̂^i_0 = G_phi(x^i_{t^i}, t^i)   # 每个 chunk 预测干净帧
    x̂_0 = [x̂^1_0, ..., x̂^L_0]      # 拼接成完整视频

    # ---- DMD 梯度计算 ----
    t ~ Uniform(0, T)               # 再采一个全局时间步
    x̂_t = α_t x̂_0 + σ_t ε         # 对 student 输出加噪
    # 用 s_data（bidirectional teacher）和 s_gen（fake score）
    # 的差作为梯度信号，更新 G_phi

    # ---- 更新 fake score s_gen ----
    x̂_t = α_t x̂_0 + σ_t ε'        # 重新加噪（独立噪声）
    # 用去噪 loss（Eq.2）在 student 生成的样本上训练 s_gen
```

**关键设计：Diffusion Forcing 的噪声方案**

第 5 行：**每个 chunk 独立采样自己的时间步** $t^i \sim \text{Uniform}(\mathcal{T})$。

这来自 Diffusion Forcing（Du et al.）的思想：训练时允许视频中不同帧处于不同的噪声水平，这样模型学会了在"部分干净 + 部分带噪"的混合条件下去噪，为推理时的自回归生成（前面 chunk 已干净，当前 chunk 纯噪）打下分布基础。

训练时 teacher 看到的是"各 chunk 独立噪声水平"的混合序列 ${x^i_{t^i}}$，而 student 推理时历史 chunk 是 $t^i=0$（干净），当前 chunk 是 $t^i = t_Q$（纯噪）。两者的噪声结构在 Diffusion Forcing 的框架下具有一定的"包含关系"——推理时的状态是训练时状态空间的一个特例，但仍然是有 bias 的特例，不是精确覆盖。

------

### 4.3 Student Initialization（ODE 回归预热）

直接从随机权重或 teacher 权重开始做 DMD 不稳定，原因是 student（causal）和 teacher（bidirectional）的架构不同，初始输出分布差异很大，导致 $s_\text{data} - s_\text{gen}$ 的梯度信号极端嘈杂。

用 bidirectional teacher 跑 ODE solver（DDIM），生成 **1000 对** $(x_t, x_0)$ 配对数据（ODE trajectory 上的点对），然后在这些配对上做 MSE 回归：
$$
\mathcal{L}_\text{init} = \mathbb{E}_{x, t^i} \left\| G_\phi\bigl(\{x^i_{t^i}\}_{i=1}^N,\; \{t^i\}_{i=1}^N\bigr) - \{x^i_0\}_{i=1}^N \right\|_2^2
$$
注意这里的 ${x^i_{t^i}}$ 是 teacher ODE 轨迹上对应 student 推理时间步的点，${x^i_0}$ 是对应的干净终点。

**这一步的作用**：让 causal student 在开始 DMD 之前，已经能大致模仿 bidirectional teacher 的 ODE 映射，输出分布合理，$s_\text{gen}$ 能提供有意义的梯度，训练更稳定。实际用量：**仅 1000 ODE 对，训练 3000 步**，计算代价极小（相对 6000 步 DMD 而言）。

------

### 4.4 KV Caching 推理

**Algorithm 2**

```python
# 推理时不再需要训练时的 block-wise causal mask
# 改用标准 bidirectional attention + KV cache，速度更快

C ← ∅  # KV cache 初始化为空

for i = 1 to L:  # 逐 chunk 生成
    x^i_{t_Q} ~ N(0, I)   # 当前 chunk 从纯高斯噪声出发

    # 4步去噪
    for j = Q downto 1:
        x̂^i_{t_j} = G_phi(x^i_{t_j}, t_j) using cache C
        x^i_{t_{j-1}} = α_{t_{j-1}} x̂^i_{t_j} + σ_{t_{j-1}} ε'

    # 当前 chunk 生成完毕（x^i_0），更新 KV cache
    KV pairs = G_phi(x^i_0, 0) 的前向计算
    C.append(KV pairs)
```

**一个关键细节**：推理时用的是 **full bidirectional attention**（利用 FlexAttention 的快速实现），通过 KV cache 复用历史帧的 K/V，不需要 block-wise causal mask。这是因为 KV cache 本身已经保证了因果性（当前帧只能 attend 到 cache 中的历史帧 K/V），而 bidirectional attention 实现比 masked attention 更高效。

------

## 整体流程总结

```
阶段 0（Pre-train）：
  Bidirectional teacher（CogVideoX-style DiT，50步）在大规模视频数据上预训练

阶段 1（ODE 初始化，3000 iter）：
  Teacher 跑 ODE → 生成 1000 对 (x_t, x_0)
  Causal student 在这些对上做 MSE 回归，学会粗略模仿 teacher

阶段 2（非对称 DMD，6000 iter）：
  - Teacher（bidirectional）冻结，作为 s_data
  - s_gen 用 teacher 初始化，在线用 student 输出更新
  - Student（causal）用 DMD 梯度（s_data - s_gen）更新
  - 各 chunk 独立采噪声水平（Diffusion Forcing）
  - 总训练：64 × H100，约 2 天

推理（4步，KV cache）：
  逐 chunk 纯噪声出发 → 4步去噪 → 存 KV cache → 下一 chunk
  初始延迟 1.3s，后续 9.4 FPS
```

------

## 与相关工作的核心差异

|                    | Bidirectional Diffusion（Sora-style） | 直接训练 Autoregressive | CausVid                                 |
| ------------------ | ------------------------------------- | ----------------------- | --------------------------------------- |
| 注意力             | Full bidirectional                    | Causal                  | Causal                                  |
| 速度               | 慢（50步 × 全序列）                   | 慢-中                   | 快（4步 + KV cache）                    |
| Error Accumulation | 无                                    | 严重                    | 通过 bidirectional teacher 蒸馏大幅缓解 |
| 长视频             | 受限于帧数                            | 可外推                  | 可外推（short clip 训练，长视频推理）   |
| 交互性             | 无                                    | 有                      | 有（streaming + dynamic prompt）        |

该方法有效缓解了自回归生成中的误差累积，使得仅在短 clip 上训练的模型也能合成长时视频。

------

## 实验结果

在 VBench-Long benchmark 上达到 84.27 的总分，超越此前所有视频生成模型。

零样本下游任务（无需 finetune）：

- Image-to-video（首帧 conditioning）
- Video-to-video translation（streaming）
- Dynamic prompting（生成过程中实时切换 text prompt）

------

## 局限性

论文自陈（Appendix C）：

- **Long-range Inconsistency**：KV cache 长度有限，超出 context window 的远距帧无法保证一致性
- **Temporal Discontinuity at Chunk Boundaries**：按 chunk 自回归时，相邻 chunk 边界处可能出现跳帧