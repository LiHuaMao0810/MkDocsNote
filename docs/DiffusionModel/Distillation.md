# 蒸馏文献整理



## 原理

[Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)

用GPT或者Gemini等大模型作为Reward Model，让模型进行自我蒸馏，将大模型的鉴赏能力蒸馏进相对较小的flow matchig model中



[Align Your Flow: Scaling Continuous-Time Flow Map Distillation](https://arxiv.org/abs/2506.14603)

从数学角度论证了一致性模型本质上和多步采样不兼容，提出了flow map(类似轨迹一致性蒸馏)，并使用了弱模型负向引导进行蒸馏



[SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](https://arxiv.org/abs/2503.09641)

对一致性蒸馏的改进，预测相邻时间步状态的时候，使用更精确近似而不是一节线性近似，减少离散化误差。最后的损失函数和一致性蒸馏一样。



[Inductive Moment Matching](https://arxiv.org/abs/2503.07565)

轨迹一致性蒸馏加分布匹配损失换了个名字



[Mean Flows for One-step Generative Modeling](https://arxiv.org/pdf/2505.13447)

训练目标为平均速度场的蒸馏



**[How to build a consistency model: Learning flow maps via self-distillation](https://arxiv.org/abs/2505.18825)** 

指出轨迹一致性蒸馏，MeanFlow，渐进式蒸馏是等价的，并给出了统一的一致性模型训练框架



SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation

解决传统DMD在FLUX等大模型上蒸馏时，训练模式容易崩溃的问题



Variational Rectified Flow Matching

在ODE方程引入引入随机项，变为SDE进行最大化ELBO蒸馏



Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency

连续时间一致性蒸馏加上Score matching作为正则项





## 应用

### 2025

[FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space](https://arxiv.org/abs/2506.15742)

论文中提到了利用 **Latent Adversarial Diffusion Distillation (LADD)** 等技术，使得这个巨型模型的推理速度比同类模型（如 GPT-4o 图像版）快约 8 倍。



[GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving](https://arxiv.org/abs/2503.05689)

在自动驾驶领域使用流匹配蒸馏产生高质量的多模态轨迹



[TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models](https://arxiv.org/abs/2502.06608)

在3D生成领域使用rectified flow来训练模型



### 2026

MTFlow: Time-Conditioned Flow Matching for Microtubule Segmentation in Noisy Microscopy Images

时间流匹配模型在细胞微管分割的应用



FMVP: Masked Flow Matching for Adversarial Video Purification

使用条件流匹配修复对抗输入视频



FlowLet: Conditional 3D Brain MRI Synthesis using Wavelet Flow Matching

使用流匹配生成年龄可调节的3D 脑磁共振成像数据



MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation based Distillation

在自动驾驶领域进行蒸馏，用隐式最大似然，要求模型的输出符合教师的一个样本点，而不是取平均
