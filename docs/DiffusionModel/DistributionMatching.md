---
tags:
  - 分数匹配
  - 蒸馏
  - 扩散模型
---

# **Distribution Matching Distillation 分布匹配蒸馏**

> [!info]
>
> 创建时间：2025-11-29 | 更新时间：2025-12-16
>
> 本文基于**[One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828)** 做笔记



## **主要贡献**

提出了分布匹配蒸馏方法，即我们不需要让模型学习每一次预测的噪声，而是在最终得到的图片上进行损失计算。

因为最终的图片也可以称作是目标数据分布，所以叫做分布匹配蒸馏？

通过最小化单步生成器和扩散模型生成分布的KL散度来优化生成器。而该 KL 散度的梯度可以表示为两个评分函数之差，其中一个评分函数对应于目标分布，另一个评分函数对应于我们单步生成器生成的合成分布。



## **核心公式**

核心的分布匹配损失写作：

![image-20251202200829817](./assets/image-20251202200829817.png)

这种概率密度损失难以估计，幸好我们只需要求他对于参数 θ 的梯度：

![image-20251202200915608](./assets/image-20251202200915608.png)

其中 $s_{\text{real}}(x) = \nabla_x \log p_{\text{real}}(x) \quad$和$\quad s_{\text{fake}}(x) = \nabla_x \log p_{\text{fake}}(x)$ 代表相应匹配的分数 Score 

使用扩散模型对分数进行建模，得到


$$
s_{\text{real}}(x_t, t) = -\frac{x_t - \alpha_t \mu_{\text{base}}(x_t, t)}{\sigma_t^2} \\
s_{\text{fake}}(x_t, t) = -\frac{x_t - \alpha_t \mu_{\text{fake}}^\phi(x_t, t)}{\sigma_t^2}.
$$


其中 $\mu_{base}$ 是学习到真实分布的基座模型， $\mu_{fake}^{\phi}$ 是我们训练的学生模型，$\alpha_t , \sigma_t$ 都是噪声调度器的参数

> [!note]
>
> 这个建模公式来源于[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) ，在[Understanding Diffusion Model](./UnderstandingDiffusionModels.md)笔记中也记录了分数和噪声预测的关系。

## **Tricks**

在训练期间，因为我们单步生成器的合成分布一直在变化，我们需要通过最小化标准的去噪目标来更新 $\phi$ 


$$
\mathcal{L}_{\text{denoise}}^\phi = \|\mu_{\text{fake}}^\phi(x_t, t) - x_0\|_2^2,
$$

且对于少量噪声的情况，$p_{real}(x_t,t)$ 趋向于0， 所以 $s_{real}(x_t,t)$ 的值不稳定，训练容易崩溃，为此需要引入额外的回归损失，这里取图像块相似性（LPIPS）

​	
$$
\mathcal{L}_{\text{reg}} = \mathbb{E}_{(z, y) \sim \mathcal{D}} \ell(G_\theta(z), y).
$$

