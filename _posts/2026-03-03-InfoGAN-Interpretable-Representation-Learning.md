---
layout: post
title: "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets"
author: Guillermo Moreno
categories: [paper-summary]
tags: [generative-adversarial-networks, information-theory, representation-learning, paper-summary]
---

Source: [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) ([PDF](https://arxiv.org/pdf/1606.03657)).

## Classic GAN

$$
\min_G\max_D V(D,G)
=\mathbb{E}_{x\sim p_{\mathrm{data}}}\left[\log D(x)\right]
+\mathbb{E}_{z\sim p_{\mathrm{noise}}}\left[\log\left(1-D(G(z))\right)\right].
$$

## Idea of Mutual Information for inducing latent codes

- $z$: source of incompressible noise.
- $c$: latent code, targeting salient and structured semantic features.

For these examples, the latent codes are assumed to be conditionally independent.

- It is necessary to impose a Mutual Information regularization to avoid the generator ignoring $c$. The dependency

  $$
  c\longrightarrow G(z,c)\longrightarrow p_G(x\mid c)
  $$

  should be strong, so $I(c;G(z,c))$ should be high.
- Note on Mutual Information (MI), $I(x;y)$:
  - measures the amount of information learned about a random variable $x$ from knowledge of a random variable $y$,
  - measures the reduction of uncertainty in $x$ when $y$ is observed,
  - if $x$ and $y$ are independent, then $I(x;y)=0$.

## Variational Mutual Information Maximization

- To compute $I(c;G(z,c))$, it is necessary to compute the posterior $P(c\mid x)$.
- Introduce $Q(c\mid x)$ to approximate $P(c\mid x)$.

$$
\begin{aligned}
I(c;G(z,c))
&=H(c)-H(c\mid G(z,c)) \\
&=\mathbb{E}_{x\sim G(z,c)}
  \left[\mathbb{E}_{c'\sim P(c'\mid x)}
  \left[\log P(c'\mid x)\right]\right]+H(c) \\
&=\mathbb{E}_{x\sim G(z,c)}
  \left[D_{\mathrm{KL}}\left(P(\cdot\mid x)\Vert Q(\cdot\mid x)\right)
  +\mathbb{E}_{c'\sim P(c'\mid x)}\left[\log Q(c'\mid x)\right]\right]+H(c) \\
&\geq \mathbb{E}_{x\sim G(z,c)}
  \left[\mathbb{E}_{c'\sim P(c'\mid x)}
  \left[\log Q(c'\mid x)\right]\right]+H(c),
\end{aligned}
$$

because

$$
D_{\mathrm{KL}}\left(P(\cdot\mid x)\Vert Q(\cdot\mid x)\right)\geq 0.
$$

### Lemma

For random variables $X,Y$ and a function $f(x,y)$, under suitable regularity conditions:

$$
\mathbb{E}_{x\sim X,\,y\sim Y\mid x}[f(x,y)]
=\mathbb{E}_{y\sim Y,\,x'\sim X\mid y}[f(x',y)].
$$

Applying the lemma to the variational mutual information estimator gives:

$$
\begin{aligned}
L_I(G,Q)
&=\mathbb{E}_{c\sim p(c),\,x\sim G(z,c)}\left[\log Q(c\mid x)\right]+H(c) \\
&\leq I(c;G(z,c)).
\end{aligned}
$$

This lower bound can be optimized efficiently by sampling $c\sim p(c)$ and $z\sim p(z)$, generating $x=G(z,c)$, and evaluating $\log Q(c\mid x)$.

## Proposed loss

Considering the original formulation of the GAN loss, the information-regularized objective is:

$$
\min_{G,Q}\max_D V_{\mathrm{InfoGAN}}(D,G,Q)
=V(D,G)-\lambda L_I(G,Q).
$$

Maximizing $L_I(G,Q)$ encourages $G$ to preserve the information encoded by $c$ in the generated sample. Therefore, changing a component of $c$ should produce a consistent and interpretable change in the generated output, while $z$ continues to model incompressible variation.
