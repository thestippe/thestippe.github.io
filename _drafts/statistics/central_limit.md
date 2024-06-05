---
layout: post
title: "The central limit theorem"
categories: /statistics/
tags: /central_limit/
subcategory: Introduction
date: "2024-01-11"
section: 6
# image: "/docs/assets/images/perception/eye.jpg"
description: "Approximating the probability when the sample size is large"
published: false
---

We have derived some of the most common probability
distribution functions.
In this post we will discuss their behavior for an iid sample
when the sample size grows.

## The weak law of large numbers

Let us consider an iid $X_1,\dots,X_n$ sample where the underlying distribution
for all the $X_i$s has finite mean $\mu$ and finite variance $\sigma^2\,,$
and let us indicate the sample average as

$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

We have

$$
\mathbb{E}[\bar{X}]  =
\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n X_i\right]  =
 \frac{1}{n}\sum_{i=1}^n \mathbb{E}[ X_i]  = \mu
$$


$$
Var[\bar{X}] =
\mathbb{E}\left[ \left(\frac{1}{n} \sum_{i=1}^n X_i - \mu \right )^2 \right ]
=
\mathbb{E}\left[ \left(\frac{1}{n} \sum_{i=1}^n (X_i - \mu) \right )^2 \right ]
=
\frac{1}{n^2}\mathbb{E}\left[ \sum_{i=1}^n (X_i - \mu)^2  \right ] = \frac{\sigma^2}{n}
$$

Above we used $\mathbb{E}[(X_i-\mu)(X_j-\mu)] = 0$ if $i\neq j\,,$ since
the variables are mutually independent.

The one above is the **weak law of large numbers**.

## The central limit theorem

Consider an iid sample of random variables $X_1,\dots,X_n\,.$
with zero mean and unit variance, and let $Z_n = \frac{1}{\sqrt{n}} \sum_i X_i$

$$
\begin{align}
\phi_{Z_n}(t) & = \mathbb{E}_{Z_n}[e^{i t \bar{X}}] = \int dx e^{i t x} p_{Z_n}(x)  \\
&
= \int dz \int dx_1 \cdots dx_n \delta\left(z - \frac{1}{\sqrt{n}} \sum_i x_i\right) e^{i t \frac{1}{\sqrt{n}}\sum_i x_i} \prod_i p_i(x_i) \\
&
=  \int dx_1 \cdots dx_n e^{i t \frac{1}{\sqrt{n}}\sum_i x_i} \prod_i p_i(x_i) \\
&
= \prod_i \phi_i(t/\sqrt{n}) = \phi_1(t/\sqrt{n})^n
\end{align}
$$

By expanding with respect to $\frac{t}{\sqrt{n}}\,,$
since $\phi(t) \approx \mathbb{E}[1] + \mathbb{E}[X] t + \frac{1}{2} \mathbb{E}[X^2] t^2+ O(t^3) = 1-\frac{t^2}{2}+O(t^3)$

$$
\phi_1(t/\sqrt{n}) \approx 1-\frac{t^2}{2n} + O\left(\frac{t^3}{n^{3/2}}\right)
$$

$$
\lim_{n \rightarrow \infty} \left(1-\frac{t^2}{2 n}\right)^n = e^{-\frac{t^2}{2}}
$$

so, in the large $n$ limit, $Z_n$ approaches a normal random variable with zero mean and unit variance.

If $X_1,\dots,X_n$ is a collection of $n$ iid random variables with finite mean $\mu$ and finite variance $\sigma\,,$
and if we define $S_n = \sum_{i=1}^n X_i = n \bar{X}\,,$ then

We can rephrase the above proof of the **central limit theorem** (CLT) as

$$
\frac{S_n - n\mu}{\sqrt{n} \sigma} = \frac{\sqrt{n}(\bar{X}-\mu)}{\sigma} \sim \mathcal{N}(0, 1)
$$

Above we introduced the notation $\sim\,,$ where we mean "is distributed according to", and $\mathcal{N}(\mu, \sigma)$
indicates the normal distribution with mean $\mu$ and variance $\sigma\,.$
We will often use this notation, introducing each time the corresponding symbol for the random distribution.

More formally, the above notation should be

$$
x | \mu, \sigma \sim \mathcal{N}(\mu, \sigma)
$$

but since the conditioning should appear clear, we will usually omit it.

## Conclusions

We proved two very important theorems for the mean of a set of iid random variables.
In the next posts we will discuss some general topic about frequentist inference.
