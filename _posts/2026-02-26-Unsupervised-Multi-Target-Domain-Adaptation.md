---
layout: post
title: "Unsupervised Multi-Target Domain Adaptation: An Information-Theoretic Approach"
author: Guillermo Moreno
categories: [paper-summary]
tags: [domain-adaptation, information-theory, representation-learning, paper-summary]
---

Source: [Unsupervised Multi-Target Domain Adaptation: An Information Theoretic Approach](https://arxiv.org/abs/1810.11547) ([PDF](https://arxiv.org/pdf/1810.11547)).

Propose an information-theoretic approach for domain adaptation in the novel context of multiple target domains with unlabeled instances and one source domain with labeled instances.

Find latent spaces common to all domains, while simultaneously accounting for remaining private, domain-specific factors.

Specifically, simultaneously factorize the information from each available target domain and learn separate subspaces for modelling the shared (correlated across domains) and private (independent between domains) data. To do this:

- jointly maximize the mutual information between the domain labels and private features (domain-specific),
- minimize the mutual information between the domain labels and the shared features (domain-invariant).

# Preliminaries

## Information theory: Background

Let $\textbf{x}=(x_1,x_2,\ldots,x_n)$ be an $n$-dimensional random variable with probability distribution $p(\textbf{x})$. The Shannon differential entropy is:

$$
H(\textbf{x})=-\langle\ln p(\textbf{x})\rangle_{\textbf{x}}.
$$

Let $\textbf{z}=(z_1,z_2,\ldots,z_m)$ be an $m$-dimensional random variable with probability distribution $p(\textbf{z})$. The mutual information between $\textbf{x}$ and $\textbf{z}$ is:

$$
\begin{aligned}
I(\textbf{x};\textbf{z})
&=H(\textbf{x})+H(\textbf{z})-H(\textbf{x},\textbf{z}) \\
&=H(\textbf{x})-H(\textbf{x}\mid\textbf{z}) \\
&=H(\textbf{z})-H(\textbf{z}\mid\textbf{x}).
\end{aligned}
$$

# Method

## Problem statement

Let

$$
(\textbf{X},\textbf{Y},\textbf{D})
=\{(\textbf{x}_i,\textbf{y}_i,\textbf{d}_i)\}_{i=1}^{N}
$$

be a collection of $M$ domains, with one labeled source domain and $M-1$ unlabeled target domains.

- $\textbf{x}_i$: the $i$-th point. For image classification, $\textbf{x}_i\in\mathbb{R}^{H\times W\times C}$, with $H,W,C$ the dimensions of the images.
- $\textbf{y}_i$: the label of the $i$-th point. For image classification, $\textbf{y}_i\in\mathbb{R}^{K}$, with $K$ classes.
- $\textbf{d}_i$: the domain label of the $i$-th point. In general, $\textbf{d}_i\in\mathbb{R}^{M}$, with $M$ domains.

A latent space with shared and private features $\textbf{z}=[\textbf{z}_s,\textbf{z}_p]$ of $\textbf{x}$ is searched by the model. With these variables, the following mappings are proposed:

- $\textbf{z}_s=E_{\theta_s}(\textbf{x},\textbf{d})$
- $\textbf{z}_p=E_{\theta_p}(\textbf{x},\textbf{d})$
- $\hat{\textbf{y}}=C_{\theta_c}(\textbf{z}_s)$
- $\hat{\textbf{d}}=D_{\psi}(\textbf{z})$

The aim is to maximize:

$$
\begin{aligned}
L(\theta_s,\theta_p,\theta_c;\textbf{x},\textbf{y},\textbf{d})
={}&\lambda_r I(\textbf{x};\textbf{z})
+\lambda_c I(\textbf{y};\textbf{z}_s) \\
&+\lambda_d\left(I(\textbf{d};\textbf{z}_p)-I(\textbf{d};\textbf{z}_s)\right).
\end{aligned}
$$

Note: a term minimizing the mutual information between $\textbf{z}_p$ and $\textbf{z}_s$ might be considered. However, computing mutual information is intractable due to the complex joint distribution $p(\textbf{z}_s,\textbf{z}_p)$.

## Optimization

A lower bound for mutual information is derived from variational methods on the KL divergence:

$$
I(\textbf{x};\textbf{z})
\geq H(\textbf{x})
+\left\langle\ln q(\textbf{x}\mid\textbf{z};\phi)\right\rangle_{p(\textbf{x},\textbf{z})}.
$$

A variational approximation $q$ is used because the computation of

$$
p(\textbf{x}\mid\textbf{z})
=\frac{p(\textbf{z}\mid\textbf{x})p(\textbf{x})}{p(\textbf{z})}
$$

is intractable. This lower bound is applied to $\textbf{d}$ and $\textbf{y}$:

$$
\begin{aligned}
I(\textbf{d};\textbf{z}_p)
&\geq H(\textbf{d})
+\left\langle\ln q(\textbf{d}\mid\textbf{z}_p;\psi)\right\rangle_{p(\textbf{d},\textbf{z}_p)}, \\
I(\textbf{d};\textbf{z}_s)
&\geq H(\textbf{d})
+\left\langle\ln q(\textbf{d}\mid\textbf{z}_s;\psi)\right\rangle_{p(\textbf{d},\textbf{z}_s)}, \\
I(\textbf{y};\textbf{z}_s)
&\geq H(\textbf{y})
+\left\langle\ln p(\textbf{y}\mid\textbf{z}_s)\right\rangle_{p(\textbf{y},\textbf{z}_s)}.
\end{aligned}
$$

A mapping for reconstruction is proposed:

- $\hat{\textbf{x}}=F_\phi(\textbf{z})$

The variational distributions are defined as:

- $\ln q_\phi(\textbf{x}\mid\textbf{z})\propto-\Vert\textbf{x}-F_\phi(\textbf{z})\Vert_1$
- $\ln q_\psi(\textbf{d}\mid\textbf{z})=\textbf{d}^{\top}\ln D_\psi(\textbf{z})$
- $\ln p(\textbf{y}\mid\textbf{z}_s)=\textbf{y}^{\top}\ln C_{\theta_c}(\textbf{z}_s)$

Adversarial training is used to optimize a minimax saddle-point problem.

### Optimizing $\phi$ of decoder

$$
\hat{\phi}=\arg\min_\phi L_F
=\frac{\lambda_r}{N}\sum_{i=1}^{N}
\left\Vert
\textbf{x}_i-F_\phi\left(E_{\theta_s}(\textbf{x}_i),E_{\theta_p}(\textbf{x}_i)\right)
\right\Vert_1.
$$

### Optimizing $\psi$ of domain classifier

$$
\begin{aligned}
\hat{\psi}=\arg\min_\psi L_D
={}&-\frac{\lambda_d}{N}\sum_{i=1}^{N}
\textbf{d}_i^{\top}\ln D_\psi(E_{\theta_s}(\textbf{x}_i)) \\
&-\frac{\lambda_d}{N}\sum_{i=1}^{N}
\textbf{d}_i^{\top}\ln D_\psi(E_{\theta_p}(\textbf{x}_i)).
\end{aligned}
$$

### Optimizing $\theta_c$ of label classifier

The classifier loss combines supervised classification on the source domain, entropy minimization on target predictions, and a batch-level term that encourages diverse target predictions. Let

$$
\textbf{p}_i=C_{\theta_c}(E_{\theta_s}(\textbf{x}_i)),
\qquad
\bar{\textbf{p}}=\frac{1}{N-N_s}\sum_{i=N_s+1}^{N}\textbf{p}_i.
$$

Then:

$$
\begin{aligned}
\hat{\theta}_c=\arg\min_{\theta_c}L_C
={}&-\frac{\lambda_c}{N_s}\sum_{i=1}^{N_s}
\textbf{y}_i^{\top}\ln\textbf{p}_i \\
&-\frac{\lambda_c}{N-N_s}\sum_{i=N_s+1}^{N}
\textbf{p}_i^{\top}\ln\textbf{p}_i \\
&+\lambda_c\,\bar{\textbf{p}}^{\top}\ln\bar{\textbf{p}}.
\end{aligned}
$$

### Optimizing $\theta_p$ of private encoder

$$
\begin{aligned}
\hat{\theta}_p=\arg\min_{\theta_p}L_p
={}&\frac{\lambda_r}{N}\sum_{i=1}^{N}
\left\Vert
\textbf{x}_i-F_\phi\left(E_{\theta_s}(\textbf{x}_i),E_{\theta_p}(\textbf{x}_i)\right)
\right\Vert_1 \\
&-\frac{\lambda_d}{N}\sum_{i=1}^{N}
\textbf{d}_i^{\top}\ln D_\psi(E_{\theta_p}(\textbf{x}_i)).
\end{aligned}
$$

### Optimizing $\theta_s$ of shared encoder

The shared encoder combines reconstruction, adversarial domain confusion, and the label-classifier objective:

$$
\begin{aligned}
\hat{\theta}_s=\arg\min_{\theta_s}L_s
={}&\frac{\lambda_r}{N}\sum_{i=1}^{N}
\left\Vert
\textbf{x}_i-F_\phi\left(E_{\theta_s}(\textbf{x}_i),E_{\theta_p}(\textbf{x}_i)\right)
\right\Vert_1 \\
&+\frac{\lambda_d}{N}\sum_{i=1}^{N}
\textbf{d}_i^{\top}\ln D_\psi(E_{\theta_s}(\textbf{x}_i)) \\
&-\frac{\lambda_c}{N_s}\sum_{i=1}^{N_s}
\textbf{y}_i^{\top}\ln\textbf{p}_i \\
&-\frac{\lambda_c}{N-N_s}\sum_{i=N_s+1}^{N}
\textbf{p}_i^{\top}\ln\textbf{p}_i \\
&+\lambda_c\,\bar{\textbf{p}}^{\top}\ln\bar{\textbf{p}}.
\end{aligned}
$$

The sign of the domain term for $E_{\theta_s}$ is reversed relative to the domain classifier's own optimization. This makes the shared features domain-invariant, while the private encoder is trained to retain domain information.
