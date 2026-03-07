# 模型架构

本模块涵盖大模型架构相关的核心知识点。

## 主要内容

### Transformer 架构

- 自注意力机制（Self-Attention）
- 多头注意力（Multi-Head Attention）
- 位置编码（Positional Encoding）
- 前馈神经网络（FFN）

### 经典模型

- **BERT**：双向编码器表示
- **GPT 系列**：自回归语言模型
- **T5**：Text-to-Text 框架
- **LLaMA**：开源大语言模型

### 架构改进

- **Flash Attention**：高效注意力计算
- **Sparse Attention**：稀疏注意力
- **Mixture of Experts (MoE)**：专家混合模型
- **长上下文扩展**：RoPE、ALiBi 等

### 常见面试问题

- Transformer 的优缺点是什么？
- BERT 和 GPT 的区别？
- 如何解决长序列建模问题？
- 注意力机制的复杂度如何优化？

## 学习路径

1. 掌握 Transformer 基础
2. 理解各类模型的设计思想
3. 了解最新的架构改进
4. 实践：从零实现简单的 Transformer

---

> [!tip] 符号约定 
>
> - **$B$ (Batch Size)：** 一次处理多少个句子（比如一次处理 4 个人的对话）。
>- **$L$ 或 $N$ (Sequence Length)：** 序列长度。一个句子有多少个 Token（比如 4096 个词）。**这是最关键的变量。**
> - **$D$ 或 $d_{model}$ (Embedding Dimension)：** 模型维度。一个词向量的长度（比如 Llama-7B 是 4096）。
> - **$H$ (Heads)：** 多头注意力的头数（比如 32 个头）。
> - **$d_k$ 或 $d_{head}$：** 每个头的维度。通常 $d_k = D / H$（比如 $4096 / 32 = 128$）。

