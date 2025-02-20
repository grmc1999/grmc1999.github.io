---
layout: post
title: Denoising Diffusion Probabilistic Models
---

Diffusion models:

$$
P_{\theta}(X_0)=\int P_{\theta}(X_{0:T}) d X_{1:T}
$$

With:

1. $$\{X\}_{1:T}$$: are latent variable of some dimensionality
2. $$X_0 \sim q(X_0);P_\theta(X_{0:T})$$: Reversed process

Learned with gaussian transition

- Start: $$P(X_T)= N(X_T;0,I)$$
- $$P_\theta(X_{0:T})=P(X_T) \prod_{t=1}^T P_\theta(x_{t-1} \mid x_t)$$
- $$P_\theta(x_{t-1} \mid x_t)=N(x_{t-1},\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$$

Forward process: $$q(x_{1:T} \mid x_0)=\prod_{t=1}^T q(x_t \mid x_{t-1})$$

with $$q(x_t \mid x_{t-1})=N(x_t;\sqrt{1-\beta_t}x_{t-1};\beta_t I)$$

Then the log-likelihood:

$$
L=\langle -\log P_\theta(x_0)\rangle=\left \langle -\log \frac{P_\theta(x_{0:T})}{q(x_{1:T} \mid x_0} \right \rangle_q= \left \langle -\log P(X_T)-\sum_{t \geq 1} \log \frac{P_\theta(X_{t-1} \mid X_t)}{q(x_t \mid x_{t-1})} \right \rangle_q
$$

is is possible to sample q from any t from the fact that: $$\alpha_t := 1-\beta_t;\bar{\alpha}:=\prod_{s=1}^t \alpha_s$$ then:

$$
q(X_t \mid X_0)=\mathcal{N} (X_t \mid \sqrt {1-\beta_t}X_{t-1};\beta_tI)
$$

With this the log-likelihood is:

$$
L=\left \langle \mathbb D_{KL}(q(X_T \mid X_0) \| P(X_T))+\sum_{t>1} \mathbb D_{KL}(q(X_{t-1} \mid X_t,X_0) \| P_\theta(X_{t-1} \mid X_t)) -\log P_\theta(X_0 \mid X_1) \right \rangle_q
$$

The loss is tractable conditioning on $$X_0$$: $$q(X_{t-1} \mid X_t,X_0)=\mathcal N (X_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I)$$

$$
q(X_{t-1} \mid X_t,X_0)=\mathcal N (X_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I)
$$

with

$$
\tilde\mu_t(X_t,X_0)=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_{t-1}} X_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}X_t
$$

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

Forward process: $$q(X_{1:T} \mid X_0)$$ has no learnable parameters

Reverse process: $$P_\theta(X_{t-1} \mid X_t)=\mathcal N (X_{t-1};\mu_\theta(X,t),\sigma^2_t I)$$ (for this especific process, which assumes $$\Sigma_\theta(X_t,t)=\sigma^2_tI$$

In the loss:

$$
L_{t-1}=\left \langle \frac{1}{2\sigma^2_t} \Vert \tilde\mu_t(X_t,X_0)-\mu_\theta(X_t,t) \Vert^2 \right \rangle_q +C
$$

This loss trains $\theta$ such that $$\mu_\theta$$ predicts $$\tilde\mu_t$$

with $$q(X_t \mid X_0)=\mathcal N(X_t;\sqrt{\bar\alpha_t}X_0,(1-\bar\alpha_t)I)$$ and $$X_t(X_0,\epsilon)=\sqrt{\bar\alpha_t}X_0+\sqrt{1-\bar\alpha_t}\epsilon$$

$$
L_{t-1}-C=\left \langle \frac{1}{2\sigma^2_t} \left \Vert\frac{1}{\sqrt{\alpha_t}} \left ( X_t(X_0,\epsilon) - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon \right ) - \mu_\theta(X_t(X_0,\epsilon),t) \right \Vert^2 \right \rangle_{X_0,\epsilon}
$$

with:

$$
\mu_\theta(X_t,t)=\tilde\mu_t\left(X_t,\frac{1}{\sqrt{\alpha_t}} X_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(X_t)\right )=\frac{1}{\sqrt{\alpha_t}}\left(X_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(X_t,t)\right)
$$

For sampling:

$$
X_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(X_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(X_t,t\right)+\sigma_tZ
$$

An alternative formula:

$$
\hat X_0=\frac{1}{\bar\alpha_t}X_t-\left(\sqrt{\frac{1}{\bar\alpha_t}-1}\right)\epsilon_\theta(X_t)
$$

\