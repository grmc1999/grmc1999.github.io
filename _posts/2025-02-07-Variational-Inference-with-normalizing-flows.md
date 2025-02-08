---
layout: post
title: Learning in directed models
---
# Note in: Variational Inference with normalizing flows
# Other important information in the paper

## Observed problems

- Under-estimation of variance of posterior
- Limited capacity of the posterior approximation can result in baised MAP

# Motivation

- Choice of function for approximate posterior distribution is one of the core problems of Variational Inference (VI)

# Capabilities of the method

- Flexible, arbitrarily complex and scalable approximate posterior distributions.
- Infinitesimal flows

# Preface

## Amortized Variational Inference

### Variational Inference

Given a Bayesian setting, approximate posterior distribution over latent variables with a function. This is, given random variable 

Given random variable: $$X,Z$$

$$
P(Z|X)=\frac{P(X|Z)P(Z)}{P(X)}
$$

In most of cases, solving the marginalization on unknown latent variables is intractable:

$$
P(X)=\int P(X,Z) dZ
$$

To tackle this problem, variational methods are used to approximate $$P(Z|X)$$, this is normally a parametrized function $q_{\theta}(X|Z)$ that is meant to approximate posterior. Using this approximation log-likelihood is simplified in a suitable form for learning.

$$
\log p_{\theta}(x)= \int \log p_{\theta}(x|z)p(z)dz \\ 
=  \int \log \frac{q_{\theta}(z|x)}{q_{\theta}(z|x)}p_{\theta}(x|z)p(z)dz \\
 = \int \log q_{\theta}(z|x) \frac{p_{\theta}(x,z)}{q_{\theta}(z|x)}dz \\ 
= \int \log q_{\theta}(z|x) \frac{p_{\theta}(x,z)}{q_{\theta}(z|x)}dz \\
= - \mathbb D_{KL}[q_{\theta}(z|x)||p_{\theta}(z)] + \left \langle \log p(x|z) \right \rangle_{q_{\theta}(z|x)}
$$

### Amortized Inference

Given a probabilistic model $p(x,y)$ for $X$ as latent variables and $Y$ as observed variables, amortized inference search for a parametrized mapping $g_{\theta}: Y \rightarrow P(X); y \mapsto q_{\theta}(x|z)$  such that $q_{\theta}(x|z)$ is near to $p(x|z)$. Particularly, this method apply complex parametrization to “integrate” the information of each $x_i$ is considered in the model, therefore is possible to train the model in batch using SVI.

This method is basically the use of MCMC to find mapping between latent variables.

## Main challenges in VI

1. Efficient computation of  $\nabla_{\phi}\left\langle \log p_{\theta}(x|z) \right\rangle_{q_{\theta}(z|x)}$
2. Choosing richiest computationally-feasible $q()$

## From Amortized variational inferece

## Reparametrization:

reparametrization of known distribution with a differentiable transformation

1. Reparametrization: reparametrization of known distribution with a differentiable transformation $z \sim N(z|\mu,\sigma^2) \iff z=\mu+\sigma\epsilon \ ; \ \epsilon \sim N(0,1)$
2. Backpropagration with montecarlo: Differentiate with respect to parameters $\phi$ of variational distribution, using monte carlo from the sample distribution $\nabla_\phi \left \langle f_\theta(z) \right \rangle_{q_\phi(z)} \iff \left \langle \nabla_\phi f_\theta(\mu+\sigma\epsilon) \right \rangle_{N\sim\epsilon(0,1)}$

# METHOD

## Inference Networks

An example of the simplest kind of neural network for amortized variational inferece is $q_\phi(z|x)=N(z|\mu_\phi,diag(\sigma_\phi^2(x)))$

### Deep latent Gaussian Models

Is general class of deep directed graphical models, each $z_l$ is independent with non-linear operations

$$
p(x,z_1,...,z_L)=p(x|f_0(z_1))\prod_{l=1}^L p(z_l|f_l(z_{l+1})) \\
p(z_L)=N(0,I)
$$

### Normalizing flows

Given $f: \mathbb R^d \rightarrow \mathbb R^d$  invertible smooth

$$
q(z)=q(z) \left | \det \frac{\partial f^{-1}}{\partial z'} \right | = q(z) \left | \det \frac{\partial f}{\partial z} \right |^{-1}
$$

Considering

$$
z_k= f_k( ... f_2 ( f_1 f_0( z_0 )) ...) \Rightarrow \log q_k(z_k)=\log q_0(z_0) - \sum_{k=1}^k \log \left | \det \frac{\partial f_k}{\partial z_{k-1}} \right |

$$

An important property of the successive transformations is the Law Of Unconscious Statistician, i.e. for Normalizing flows, expectations of $q_k$ can be computed without explictly knowing $q_K:\langle h(z) \rangle_{q_k} = \langle f_k( ... f_2 ( f_1 f_0( z_0 )) ...) \rangle_{q_0}$ 

The sequence of contractions and expansions allow more expressiveness, starting from a simple distribution.

### Inference with normalizing flows

 Specify a class of invertible transformations. Preferably this have a efficient mechanism to compute the determinant of the jacobian.

Compute gradient of a jacobian is unstable due matrix inverses.

for $f(z)=z+uh(w^\top z+b); w \in \mathbb R^D, u \in \mathbb R^D, b \in \mathbb R$ log-det-jacobian is O(D) and $h( ·)$ a smooth element-wise non-linearity with derivative $h’(·)$

using matrix determinant lemma:

$\left | \det \frac{\partial f}{\partial z} \right | =  \left | \det (I+u\psi(z)^\top)  \right | = |1+u^\top \psi(z)|$

The application to the initial distribution is:

$$
\log q_k(z_k) = \log q_0(z)-\sum_{k=1}^K \log |1+u_k^\top \psi_k(z_{k-1})|
$$

### Flow-based free energy bound

$$
\langle \log q_{\phi}(z|x) - \log p(x,z)\rangle_{q_\phi(z|x)} = \langle \log q_k(z_k) - \log p(x,z_k)\rangle_{q_0(z_0)} \\
= \langle \log q_k(z_k) \rangle_{q_0(z_0)}- \langle \log p(x,z_k) \rangle_{q_0(z_0)} - \langle \sum_{k=1}^K \log |1+u_k^\top\psi_k(z_{k-1})|\rangle_{q_0(z_0)}
$$
