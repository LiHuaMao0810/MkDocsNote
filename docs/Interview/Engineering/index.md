# 工程与优化

本模块聚焦大模型训练和部署中的工程实践与优化技术。

## 主要内容

### 分布式训练
- **数据并行**（Data Parallelism）
- **模型并行**（Model Parallelism）
- **流水线并行**（Pipeline Parallelism）
- **张量并行**（Tensor Parallelism）
- **Zero Redundancy Optimizer（ZeRO）**

### 模型压缩
- **知识蒸馏**（Knowledge Distillation）
- **模型剪枝**（Pruning）
- **量化**（Quantization）
  - PTQ（Post-Training Quantization）
  - QAT（Quantization-Aware Training）
- **低秩分解**（Low-Rank Factorization）

### 推理优化
- **KV Cache 优化**
- **批处理与连续批处理**
- **PagedAttention**
- **Speculative Decoding**
- **模型部署框架**（TensorRT、vLLM、TGI）

### 训练优化
- **混合精度训练**（Mixed Precision）
- **梯度累积**（Gradient Accumulation）
- **梯度检查点**（Gradient Checkpointing）
- **优化器选择**（Adam、AdamW、Lion）

### 常见面试问题
- 如何进行大模型的分布式训练？
- ZeRO 的三个阶段分别做了什么？
- 量化技术如何降低模型大小？
- KV Cache 是什么，如何优化？
- Flash Attention 为什么快？

## 学习路径

1. 理解分布式训练的基本概念
2. 掌握常用的优化技术
3. 了解主流的部署框架
4. 实践：搭建简单的训练和推理流程
