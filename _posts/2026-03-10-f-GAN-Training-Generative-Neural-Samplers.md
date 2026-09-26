---
layout: post
title: "f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization"
author: Guillermo Moreno
categories: [paper-summary]
tags: [generative-adversarial-networks, f-divergence, variational-inference, generative-models, paper-summary]
---

Source: [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/abs/1606.00709) ([PDF](https://arxiv.org/pdf/1606.00709)).

# Motivation

A desired generative model would have good properties for:

- Sampling
- Estimation
- Point-wise likelihood

In the analysis of GAN training, assuming the optimal discriminator model, the generator further optimizes the Jensen--Shannon divergence:

$$
\begin{aligned}
C(G)=V(D_G^*,G)
&=-\log(4)
+D_{\mathrm{KL}}\left(p_d\middle\Vert\frac{p_d+p_g}{2}\right)
+D_{\mathrm{KL}}\left(p_g\middle\Vert\frac{p_d+p_g}{2}\right) \\
&=-\log(4)+2D_{\mathrm{JS}}(p_d\Vert p_g).
\end{aligned}
$$

## The f-divergence family

Generalizing the divergence function, consider distributions $$P$$ and $$Q$$ with absolutely continuous density functions $$p$$ and $$q$$ defined on the domain $$\mathcal{X}$$:

$$
D_f(P\Vert Q)
=\int_{\mathcal{X}}q(x)
f\left(\frac{p(x)}{q(x)}\right)\,dx,
\qquad
f:\mathbb{R}_+\rightarrow\mathbb{R}.
$$

The function $$f:\mathbb{R}_+\rightarrow\mathbb{R}$$ is:

- convex,
- lower-semicontinuous,
- such that $$f(1)=0$$.

Different choices of $$f$$ recover different divergence measures, including the KL, reverse-KL, Jensen--Shannon, squared Hellinger, and Pearson divergences.

## Variational Divergence Minimization (VDM)

Generative-adversarial training can be stated as a special case of the Variational Divergence Minimization method. This extends the estimation of divergence measures from samples only to the estimation of generative-model parameters.

By the Fenchel conjugate, every lower-semicontinuous convex function has a convex conjugate:

$$
f^*(t)=\sup_{u\in\operatorname{dom}f}\left\{ut-f(u)\right\}.
$$

The function $$f^*$$ is convex and lower-semicontinuous, and the pair $$(f,f^*)$$ is dual. If $$f^{**}=f$$, then:

$$
f(u)=\sup_{t\in\operatorname{dom}f^*}\left\{tu-f^*(t)\right\}.
$$

Applying this representation to the divergence gives:

$$
\begin{aligned}
D_f(P\Vert Q)
&=\int_{\mathcal{X}}q(x)
\sup_{t\in\operatorname{dom}f^*}
\left\{t\frac{p(x)}{q(x)}-f^*(t)\right\}\,dx \\
&\geq\sup_{T\in\mathcal{T}}
\left(
\int_{\mathcal{X}}p(x)T(x)\,dx
-\int_{\mathcal{X}}q(x)f^*(T(x))\,dx
\right) \\
&=\sup_{T\in\mathcal{T}}
\left(
\left\langle T(x)\right\rangle_P
-\left\langle f^*(T(x))\right\rangle_Q
\right).
\end{aligned}
$$

Here, $$\mathcal{T}$$ is an arbitrary class of functions $$T:\mathcal{X}\rightarrow\mathbb{R}$$. Equality is attained when $$\mathcal{T}$$ contains the optimal function:

$$
T^*(x)=f'\left(\frac{p(x)}{q(x)}\right).
$$

To implement this approach in the GAN framework, two models are used: a generative model $$Q_\theta$$ and a variational function $$T_\omega$$. The f-GAN objective is:

$$
\min_\theta\max_\omega F(\theta,\omega)
=\min_\theta\max_\omega
\left\{
\left\langle T_\omega(x)\right\rangle_P
-\left\langle f^*(T_\omega(x))\right\rangle_{Q_\theta}
\right\}.
$$

For implementation of the objective with f-divergences, it is necessary to respect the domain of $$f^*$$. Therefore, the variational function takes the form:

$$
T_\omega(x)=g_f(V_\omega(x)),
$$

where $$V_\omega:\mathcal{X}\rightarrow\mathbb{R}$$ and $$g_f:\mathbb{R}\rightarrow\operatorname{dom}f^*$$. The function $$g_f$$ can be understood as an output activation function chosen for the divergence. The objective rewrites as:

$$
F(\theta,\omega)
=\left\langle g_f(V_\omega(x))\right\rangle_P
-\left\langle f^*\left(g_f(V_\omega(x))\right)\right\rangle_{Q_\theta}.
$$

The standard GAN objective is recovered by choosing the Jensen--Shannon divergence and its corresponding conjugate and activation function. The framework therefore generalizes adversarial training to a broader family of statistical divergences.
