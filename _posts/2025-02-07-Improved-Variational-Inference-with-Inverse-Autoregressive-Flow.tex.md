---
layout: post
title: Learning in directed models
---

# Notes on: Improved Variational Inference with Inverse Autoregressive Flow

Inverse Autoregressive Flows scale well in high dimensional latent spaces
For this method: Gaussian Autoregressive functions

## Remark Variational Autoencoder (VAE):

$$
\log p(x) \geq \langle \log p(x,z) - \log q(z|x) \rangle_{q(z|x)} = L(x,\theta) \\
L(x,\theta) =\log p(x)-\mathbb D_{KL}(q(z|x)\Vert p(z|x))
$$

Idea of context in latent variable: $$q(z_a,z_b|x)=q(z_a,x)q(z_b|z_a,x)$$

## COMPUTATIONAL INTRACTABILITY

1. Computationally efficient to compute and differentiate $$q(z|x)$$
2. Computationally efficient to sample

## Remark Normalizing Flows (NF):

Start with simple, computationally efficient distribution and apply invertible parametrized transformation $$f_t$$

$$
z_0 \sim q(z_0|x),z_t=f_t(z_{t-1},x) \forall t=1,...,T \\
\log q(z_T|x)=\log (z_0|x)- \sum^T_{t=1} \log \det \left \vert \frac{d z_t}{d z_{t-1}} \right \vert
$$

Originally: $$f(z_{t-1}=z_{t-1}+uh(w^\top z_{t-1} +b)$$

## INVERSE AUTOREGRESSIVE TRANSFORMATION

Jacobian is lower triangular

Sampling process of $y$: $y_0=u_0+\sigma_0 \odot \epsilon_0 ;y_i=\mu_i(y_{1:i-1})+\sigma_i(y_{1:i-1})\odot\epsilon_i$ for $i>0$

Inverse operation is defined: $\epsilon_i=\frac{y_i-\mu_i (y_{1:i-1})}{\sigma_i(y_{1:i-1})}$ then $\log \det \left \vert \frac{d \epsilon}{d y} \right \vert = \sum_{i=1}^D -\log \sigma_i(y)$

## Architecture

1. Perform deterministically bottom-up
2. Sampling posterior top-down

## Algorithm IAF

1. Input: $x$: data point, $\theta$: model parameters, Encoder definition: $f_\theta(x)$, Autoregressive definition $g_\theta^t(z,h)$
2. $[\mu,\sigma,h] \leftarrow f_\theta(x)$
3. $\epsilon \sim \mathcal N(0,I)$
4. $l \leftarrow - \left \Vert \log \sigma + \frac{1}{2} \epsilon^2 + \frac{1}{2} \log (2 \pi) \right \Vert_\infty$
5. for $t=1:T$:
    1. $[m,s]\leftarrow g_\theta^t(z,h)$
    2. $\sigma \leftarrow \sigma(s)$
    3. $z \leftarrow \sigma \odot z + (1-\sigma) \odot m$
    4. $l \leftarrow l - \Vert \log \sigma \Vert_\infty$
