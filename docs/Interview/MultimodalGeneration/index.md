# 多模态大模型

本模块探讨多模态大模型理论基础与前沿技术。

## 主要内容

### 视觉表征基础

*多模态的第一步是让模型“看懂”图像。*

- **ViT (Vision Transformer)：** 彻底理解图像如何 Patch 化、Linear Projection、以及 `[CLS]` Token 的作用。
- **MAE (Masked Autoencoders)：** 学习视觉领域的“自监督预训练”，理解如何通过遮盖图像块来让模型学习视觉特征。
- **重点关注：** 为什么图像需要 **2D Position Embedding**？为什么 ViT 在大规模数据下才强于 CNN？

### 对齐技术

*这是多模态的灵魂，面试 50% 的问题都在这里。*

- **CLIP (Contrastive Language-Image Pretraining)：** 深度拆解双塔架构、对比学习损失函数（InfoNCE）、以及大规模图文对齐的意义。
- **BLIP / BLIP-2 (Q-Former)：** 学习如何通过一个可学习的 Query Transformer 把海量的视觉特征“压缩”成 LLM 能听懂的几个 Token。
- **重点关注：** 对比学习中的**正负采样策略**；如何解决图文数据中的噪声问题。

### 生成架构演进

*利用你已有的扩散模型知识，完成架构迁移。*

- **DiT (Diffusion Transformer)：** 重点分析 **Adaptive LayerNorm (adaLN)** 和 **Cross-Attention** 在注入条件（如文本、时间步）时的差异。
- **Stable Diffusion 3 / Flux (Flow Matching + Transformer)：** 你已经熟悉 FM，现在要看它如何在 Transformer 架构上实现超高画质生成。
- **重点关注：** 为什么 Transformer 架构比 UNet 更适合多模态生成？（提示：参数扩展性 Scaling Law）。

### 视觉语言大模型 （VLM）

*这是目前最火的工程实践，即“给 GPT 装上眼睛”。*

- **LLaVA / InstructBLIP：** 理解 **Projection Layer**（线性投影层）的极简美学——如何只训练一个矩阵就打通视觉和文本。
- **Multimodal Instruction Tuning：** 学习如何构造“图片+指令”的数据对，让模型不仅能看图，还能听懂复杂的人类指令。
- **重点关注：** 视觉 Token 数量爆炸问题（一张图 256 个 Token 怎么优化？）。

### 可控生成与视频前沿

- **ControlNet / IP-Adapter：** 研究如何在不破坏预训练模型的情况下，注入几何控制（Canny/Depth）或风格控制。
- **Sora / Video Generation：** 学习时序注意力（Temporal Attention）或 3D 卷积，理解视频生成如何保持时空一致性。

