---
layout: post
title:  Implicit Generation and Modeling with Energy-Based Models
---

# Notes on: Implicit Generation and Modeling with Energy-Based Models


# Contributions

1. Escalable algorithms for EBM
2. Show compositionality, decorruption and inpainting
3. Useful for: out of distribution classification, adversarially robust class, multi-step trajectory prediction and online learning

# Definitions

Energy is represented as a composition of latent and observable variables

Lavengin Dynamics is a gradient based MCMC

# Energy-Based Models Sampling

$$X \in \mathbb{R}^D$$ for $$D$$ the dimension of original data, $$E_{\theta}(x) \in \mathbb \R$$ with $$E_{\theta}: \mathbb \R^D \rightarrow \mathbb \R$$ a function parametrized by trainable parameters $\theta$, as a neural network

$$
P_{\theta}(x) = \frac{\exp (- E_{\theta}(x))}{Z(\theta)} \\
Z(\theta)=\int \exp(-E_{\theta}(x)) dx
$$

$P_{\theta}$ is hard to sample. Lavengin Dynamics is used for sampling efficiently

$$
\tilde{x}^k= \tilde{x}^{k-1}-\frac{\lambda}{2}\nabla_xE_{\theta}(\tilde{x}^{k-1})+w^k;w^k \sim N(0, \lambda) \\
\tilde{x}^k \sim q_\theta; k \rightarrow \infty;\lambda \rightarrow 0 \Rightarrow q_\theta \rightarrow p_\theta
$$

Where $$E_{\theta}$$ has a “deconvolution” architecture, in this case is implemented as the gradient of the architecture.

# Maximum likelihood training

$$
\min_\theta L_{ML}(\theta)=\langle - \log P_\theta(x) \rangle_{P_D} \\
L_{ML}(\theta)= E_\theta(x) - \log(Z(\theta))\\
L_{ML}(\theta)= \langle \nabla_\theta E_\theta(x^+) \rangle_{x^+ \sim P_D} - \langle \nabla_\theta E_\theta (x^-) \rangle_{x^- \sim P_\theta}
$$

Lack of gradient is important, it controls diversity on likelihood and model collapse

Sample Replay Buffer: Saves generated $$\tilde{x}$$ to initialize MCMC Lavengin dynamics, with 95% from Buffer and 5% with noise

# Pseudo-Algorithm

## Energy training algorithm

1. Input: Data $$P_D(x)$$, Stepsize: $$\lambda$$, number of steps $$K$$
2. $$B \leftarrow \{\}$$
3. While not converge
    1.  sample $$x^+_i \sim P_D$$
    2. sample $$x^0_i \sim B$$ with 95% and $$x^0_i \sim U$$ 5%
    3. for $$k=1:k$$ : (*Generate samples with Langevin dynamics)*
        1. $$\tilde{x}^k \leftarrow \tilde{x}^{k-1} - \nabla_\theta E_\theta (\tilde{x}^{k-1} ) + w ; w \sim N(0,\sigma)$$
    4. $$x^-_i=\Omega(\tilde{x_i^k})$$ #no grad
    5. $$\Delta \theta \leftarrow \nabla_\theta \frac{1}{N}\sum\alpha(E_\theta(x^+_i)^2+E_\theta(x^-_i)^2)+ E_\theta(x^+_i)-E_\theta(x^-_i) \iff \min_\theta (\alpha L_2 + L_{ML})$$  (*Optimize Objective*)
    6. Update $\theta$ with $$\Delta \theta$$ using an optimizer
    7. $$B\leftarrow B \cup \tilde{x}_i$$