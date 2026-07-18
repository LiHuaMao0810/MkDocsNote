# 生成模型架构

---

## 1 Transformer 基础

| 文件             | 核心内容                                                     |
| ---------------- | ------------------------------------------------------------ |
| `Transformer.md` | 整体架构 / Self-Attention / FFN / Cross-Attention / 残差与归一化 / 位置编码 / 数据流 / Shape 推导 / 三种架构权衡 |

- 手推 Multi-Head Attention 的完整 Shape 变化
- 解释 Encoder-only / Decoder-only / Encoder-Decoder 的适用场景
- 说清楚 Cross-Attention 如何把文字注入图像生成过程

------

## 2 LLM 架构深化

> 从 Transformer 出发，补全工程落地和现代 LLM 的具体设计选择。

### 2.1 注意力机制优化

**文件：`attention-variants.md`（待写）**

| 话题                     | 学习重点                                                     |
| ------------------------ | ------------------------------------------------------------ |
| Flash Attention          | IO-aware 计算：为什么重新排列计算顺序能省显存，Tiling 的直觉 |
| GQA / MQA                | Group Query Attention：KV 头数 ≠ Q 头数，减少 KV Cache，LLaMA 3 用的就是这个 |
| Sliding Window Attention | Mistral 的局部注意力，如何处理超长序列                       |
| Sparse Attention         | BigBird / Longformer 的稀疏模式，$O(L)$ vs $O(L^2)$          |

**面试高频：** Flash Attention 为什么快？GQA 和 MHA 的参数量差异？

------

### 2.2 位置编码深化

**文件：`positional-encoding.md`（待写）**

| 话题           | 学习重点                                                     |
| -------------- | ------------------------------------------------------------ |
| Sinusoidal     | 原理 + 为什么相邻位置更相似 + 超长序列的退化问题             |
| 可学习位置编码 | BERT 的方案，训练长度之外的外推问题                          |
| RoPE           | 旋转矩阵推导 + 相对位置的数学性质 + YaRN / LongRoPE 的外推方法 |
| ALiBi          | 直接在 Attention 分数上加线性偏置，无需改变向量              |

**面试高频：** RoPE 如何做到只依赖相对位置？如何把上下文扩展到 128k？

------

### 2.3 归一化与训练稳定性

**文件：`normalization.md`（待写）**

| 话题                   | 学习重点                                             |
| ---------------------- | ---------------------------------------------------- |
| LayerNorm vs BatchNorm | 归一化维度的本质区别，NLP 选 LN 的原因               |
| Pre-Norm vs Post-Norm  | 梯度流动的差异，Pre-Norm 为何不需要 Warmup           |
| RMSNorm                | 去掉均值中心化，只保留缩放，LLaMA 用的方案，更省计算 |

------

### 2.4 FFN 变体与 MoE

**文件：`ffn-moe.md`（待写）**

| 话题               | 学习重点                                                     |
| ------------------ | ------------------------------------------------------------ |
| SwiGLU             | $x \cdot \text{Swish}(W_1 x) \odot W_2 x$，门控机制代替固定激活，LLaMA 默认配置 |
| Mixture of Experts | 每个 token 只激活 K 个专家，参数量大但计算量不变，Mixtral / GPT-4 的核心 |
| MoE 路由机制       | Top-K 路由 + 负载均衡 Loss，如何防止所有 token 都选同一个专家 |

**面试高频：** MoE 的参数量和计算量分别是怎么算的？

------

### 2.5 KV Cache 与推理优化

**文件：`kv-cache.md`（待写）**

| 话题                 | 学习重点                                                     |
| -------------------- | ------------------------------------------------------------ |
| KV Cache 原理        | 自回归生成中为什么可以缓存，节省了哪些重复计算               |
| 显存占用分析         | Cache 大小 = $2 \times B \times L \times H \times d_k \times \text{层数}$，如何估算 |
| Paged Attention      | vLLM 的核心：把 KV Cache 按页管理，支持动态长度              |
| Speculative Decoding | 用小模型草稿 + 大模型验证，提升吞吐量                        |

------

## 3 文生图核心组件

### 3.1 VAE（变分自编码器）

| 话题                   | 学习重点                                                     |
| ---------------------- | ------------------------------------------------------------ |
| 为什么要压缩到潜空间   | $512\times512$ 像素直接做 Attention 的计算量，Latent Diffusion 的动机 |
| Encoder / Decoder 结构 | 下采样 8 倍：$512\times512\times3 \to 64\times64\times4$     |
| KL 散度正则项          | 为什么不用普通 AE，隐变量需要满足什么分布                    |
| VQ-VAE                 | 离散潜空间变体，DALL-E 1 / AudioCodec 的基础                 |

**在 SD 里的位置：** 图像 → VAE Encoder → 潜变量（扩散/去噪在这里进行）→ VAE Decoder → 图像

------

### 3.2 扩散过程（DDPM）

| 话题                | 学习重点                                                     |
| ------------------- | ------------------------------------------------------------ |
| 前向过程            | 马尔可夫链加噪，$q(x_t | x_{t-1})$，任意时刻 $t$ 的闭合公式 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ |
| 反向过程            | 网络预测噪声 $\epsilon_\theta$，贝叶斯反转的直觉             |
| 训练目标            | 简化的 MSE Loss，为什么预测 $\epsilon$ 比预测 $x_0$ 更稳定   |
| DDIM                | 从随机采样到确定性采样，加速推理步数从 1000→20               |
| CFG（无分类器引导） | 条件生成 + 无条件生成的线性插值，$w$ 参数控制文字对图像的控制强度 |

**在 SD 里的位置：** 纯噪声 $z_T$ → 去噪网络（UNet/DiT）× $T$ 步 → 清晰潜变量 $z_0$

------

### 3.3 文本编码器（CLIP / T5）

**文件：`text-encoder.md`**

| 话题            | 学习重点                                                     |
| --------------- | ------------------------------------------------------------ |
| CLIP 对比训练   | 图文对的相似度最大化，训练目标和 InfoNCE Loss                |
| CLIP 文本塔结构 | Encoder-only，输出 shape `[seq_len, 768]`，作为 Cross-Attention 的 K/V |
| T5-XXL          | 只用 Encoder 侧，输出 `[seq_len, 4096]`，更强的长句语义理解  |
| SDXL 的双编码器 | CLIP-L + OpenCLIP-G 拼接，`[seq_len, 2048]`                  |
| FLUX / SD3      | T5-XXL + CLIP-L 同时使用，两者特征如何融合                   |

**在 SD 里的位置：** Prompt → 文本编码器 → `[L_text, D_text]` → Cross-Attention 的 K/V

------

### 3.4 去噪网络架构（UNet vs DiT）

**文件：`denoising-network.md`**

| 话题                         | 学习重点                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| UNet 结构                    | 编解码对称 + Skip Connection，ResBlock + Self-Attention + Cross-Attention 的位置 |
| 时间步条件注入               | $t$ 经过 Sinusoidal 编码后，通过 AdaLN 注入每个 Block        |
| DiT（Diffusion Transformer） | 用 Transformer 块替代 UNet，Patch Embedding 把图像切成序列   |
| MM-DiT（FLUX / SD3）         | 文字和图像 token 在同一个注意力层里互相看，双流架构          |
| ControlNet                   | 在 UNet/DiT 的基础上加旁路控制，Canny / Depth / Pose 条件如何注入 |

------

## 模块四：系统串联

> 把所有组件拼起来，理解完整的训练和推理流程。

### 4.1 Stable Diffusion 完整架构

**文件：`stable-diffusion.md`**

从一张图说清楚：Prompt → CLIP/T5 → KV；噪声 → VAE 潜空间 → UNet（Cross-Attention）→ 去噪 → VAE Decoder → 图像。每个箭头对应一个模块，每个模块的 shape 是什么。

------

### 4.2 进阶话题

| 话题                           | 文件                          |
| ------------------------------ | ----------------------------- |
| LoRA：低秩适配的数学原理       | `lora.md`                     |
| ControlNet：旁路控制的结构设计 | `controlnet.md`               |
| Flash Attention 实现细节       | 归入 `attention-variants.md`  |
| LLM 对齐：RLHF / DPO           | `alignment.md`                |
| 长上下文：YaRN / LongRoPE      | 归入 `positional-encoding.md` |

------

## 推荐学习顺序

```
Transformer.md (✅)
    ↓
diffusion.md        ←── 理解扩散过程主线
    ↓
vae.md              ←── 理解为什么在潜空间操作
    ↓
text-encoder.md     ←── 理解文字如何变成 K/V 向量
    ↓
denoising-network.md ←── UNet 和 DiT 的架构差异
    ↓
stable-diffusion.md ←── 全链路串联，复盘每个 shape
    ↓
attention-variants.md   ←── 回头补 LLM 工程细节
kv-cache.md
ffn-moe.md
```

> [!tip] 学习策略 每个文件的目标：**直觉先行 → 公式推导 → Shape 手算 → 在文生图/LLM 里的具体落点**。遇到陌生公式先问"它在做什么"，再问"它的 shape 是什么"。
