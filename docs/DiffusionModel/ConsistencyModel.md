# Consistency Model 一致性模型

!!! note
    
    




$$
\mathcal{L}_{\text{TCD}}^N(\theta, \theta^-; \phi) := \mathbb{E}\left[ \omega(t_n,t_m) \left\| f_\theta(x_{t_{n+k}}, t_{n+k},t_{m}) - f_{\theta^-}(\hat{x}_{t_n}^{\phi,k}, t_n,t_m) \right\|_2^2 \right]
$$

## 符号约定

1. $f_\theta$  参数化的一致性映射函数，接受当前时间步和隐状态，可以将当前隐状态映射到轨迹上任何一个隐状态
2. $f_{\theta^-}$ ？教师参数动态EMA更新结果，也可以直接使用 $\theta$ 
3. $\phi$ 教师流匹配去噪更新的参数
4. k  去噪步数
5. $\omega$ 时间权重函数，文中说简化为恒等于1时效果比较好



## 简化的目标函数

$$
\mathcal{L}_{\text{TCD}}^N(\theta; \phi) := \mathbb{E}\left[ \left\| f_\theta(x_{t_{n+k}}, t_{n+k},t_{m}) - f_{\theta}(\hat{x}_{t_n}^{\phi,k}, t_n,t_m) \right\|_2^2 \right]
$$

![image-20251010142826278](./assets/image-20251010142826278.png)

训练过程：

学生模型参数 $\theta$  助教模型参数 $\theta ^ -$

1. 教师模型多步采样得到一个x,视为从D采样结果

2. schedular设置timestep，从中随机选三个时间步 $t_{n+k},t_n,t_m$ (1000到0的整数)

3. 对x进行加噪，得到 $x_{t_{n+k}} = \frac{t_{n+k}}{1000} * noise + \frac{1000-t_{n+k}}{1000} *x$

4. 教师模型对x多步去噪直到t_n，得到 $\hat{x}_{t_n}^{\phi,k}$

5. 学生模型和助教模型分别从 $x_{t_{n+k}},x_{t_n}$一步去噪到$x_{t_{m}}$ 
   $$
   x_{t_{m}}= x_{t_{n+k}} + noise\_pred * (t_{n+k}-t_m)/1000
   $$

6. 二者进行MSE损失回传，更新 $\theta$。

7. with torch.no_grad: 用ema更新 $\theta ^-$

