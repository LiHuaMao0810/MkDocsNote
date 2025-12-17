---
tags:
  - 流匹配
  - 条件流匹配
  - 蒸馏
---

# **Rectified Flow 修正流**

> [!INFO] 文档信息
>
> 创建时间：2025-11-29 | 更新时间：2025-11-29
> 
> 本文基于**[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)** 做笔记



## **主要贡献**

证明了Conditional Flow Matching 在 given 初始样本 $X_0$ 的情况下也能正确工作

剩余部分提出了一些不同的前向过程，其实就是不同调度器？小巧思。

相关链接 ：[Flow Matching Guide](./FlowMatching.md) 给出了 given 目标样本 $X_1$ 下的条件流匹配证明



## **核心公式**

$$
u_t(z) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} u_t(z \mid \epsilon) \frac{p_t(z \mid \epsilon)}{p_t(z)}
$$

构造了给定初始样本时，条件路径如何聚合得到边际路径

> [!tip]
>
> 证明在文章最后

## **推导过程**

首先做一些流匹配的常规假设：

> [!INFO] 符号约定
>
> 这个文章中，$x_0$ 是目标分布的干净样本，$\epsilon$ 是给定的条件，服从高斯分布的初始噪声样本

> [!TIP] 结论前置
>
> 证明的思路就是说明给定源分布样本作为条件，得到的条件速度场，加权得到的边际速度场也可以生成所需要的边际分布 $p_t(z)$ ， 然后对损失函数进行一些重参数化，将网络预测的速度场等价转换为预测源分布样本。
>
> 实际上在这个条件流匹配的背景下，源分布样本和速度场本来就是线性变换的关系。
> t时刻的隐状态，由给定的噪声 $X_0 $:
$$
z_t = a_t x_0 + b_t \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I).
$$


隐状态边缘分布是不同 $X_0 $ 下的期望：
$$
p_t(z_t) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} p_t(z_t \mid \epsilon),
$$
用流匹配的公式表达，定义流映射和速度场：

![image-20251130170454541](./assets/image-20251130170454541.png)

**定义条件速度场如何生成边际速度场：**
$$
u_t(z) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} u_t(z \mid \epsilon) \frac{p_t(z \mid \epsilon)}{p_t(z)}
$$


用流匹配的损失函数计算

$$
\mathcal{L}_{FM} = \mathbb{E}_{t, p_{t}(z)} \| v_\theta(z, t) - u_t(z) \|^2.
$$


和Flow Matching Guide中一样的思路，直接获取这个速度场u是困难的，使用条件速度场进行替换

$$
\mathcal{L}_{CFM} = \mathbb{E}_{t, p_{t}(z \mid \epsilon), p(\epsilon)} \| v_\theta(z, t) - u_t(z \mid \epsilon) \|^2,
$$

利用


$$
\begin{aligned}
z_t' &= \psi_t'(x_0 \mid \epsilon) = a_t' x_0 + b_t' \epsilon \\
x_0 &= \psi_t^{-1}(z \mid \epsilon) = \frac{z - b_t \epsilon}{a_t}
\end{aligned}
$$



代入得：
$$
z_t' = u_t(z_t \mid \epsilon) = \frac{a_t'}{a_t} z_t - \epsilon b_t \bigg( \frac{a_t'}{a_t} - \frac{b_t'}{b_t} \bigg).
$$

引入信噪比 $\lambda_t := \log \frac{a_t^2}{b_t^2}$，且 $\lambda_t' = 2 \bigg( \frac{a_t'}{a_t} - \frac{b_t'}{b_t} \bigg)$

$$
u_t(z_t \mid \epsilon) = \frac{a_t'}{a_t} z_t - \frac{b_t}{a_t} \frac{\lambda_t'}{2} \epsilon.
$$

对公式进行重参数化得到：

![image-20251130151021069](./assets/image-20251130151021069.png)

其中定义 $\epsilon_\theta := \frac{-2}{\lambda_t' b_t} \bigg( v_\theta - \frac{a_t'}{a_t} z \bigg).$

于是网络预测就从速度场等价变为原样本了

## **边际速度场构造合理性的证明**

> [!note] 
>
> 根据lipman等人在[Flow Matching For General Modeling](https://openreview.net/forum?id=PqvMRDCJT9t)中证明的，**边际速度场生成对应边际概率，等价于其满足连续性方程**
>
> ![image-20251130170048835](./assets/image-20251130170048835.png)
>
> 第二个等号成立是因为我们设计的条件速度场能生成对应条件路径，第三个等式是假设微分和期望能互换，然后由边际分布 $p_t(z)$ 得到的

 

