---
layout: post
title: "Introduction to Gaussian processes"
categories: /statistics/
subcategory: "Time series"
tags: /gaussian_processes/
date: "2024-02-17"
# image: "/docs/5ssets/images/perception/eye.jpg"
description: "Flexible time series modelling"
section: 1
---

Gaussian processes are an extremely powerful family of models
which are based on the infinite-dimensional generalization of the
normal distribution.

We define a **Gaussian process** a collection of random variables,
any finite number of which have a joint Gaussian distribution.

As the multivariate Gaussian distribution is completely specified for any $\mathbf{x}$
by its mean and its covariance matrix, the GP is
completely defined for any real process $f(\mathbf{x})$ by
- its mean $$m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$$
- its covariance function $$k(\mathbf{x}, \mathbf{x}') = \mathbb{E}[(f(\mathbf{x})-m(\mathbf{x}))(f(\mathbf{x}')-m(\mathbf{x}'))]$$

Its definition follows immediately by the assumption that every joint
distribution follows the corresponding gaussian distribution.
Given a set of observations $$(\mathbf{x}_{i*}, y_{i*})_{i=1}^N$$, the distribution
for a new set of observation $y_j$ with $\mathbf{x}=\mathbf{x}_{j},j=1,...,M$ obeys

$$
\begin{bmatrix}
Y_* \\
Y
\end{bmatrix}
\sim
\mathcal{N}
\left(
\begin{bmatrix}
m(\mathbf{X}_*) \\
m(\mathbf{X}) 
\end{bmatrix},
\begin{bmatrix}
k(\mathbf{X}_*, \mathbf{X}_*) & k(\mathbf{X}, \mathbf{X}_*)\\
k(\mathbf{X}_*, \mathbf{X}) & k(\mathbf{X}, \mathbf{X})\\
\end{bmatrix}
\right)
$$

where $$\mathbf{X}_*=(\mathbf{x}_{i*})_{i=1}^N$$
and $$\mathbf{X}=(\mathbf{x}_{j})_{j=1}^M\,.$$

Since the kernel function $k$ must define a covariance matrix,
it must satisfy some requirement:
- it must be non-negative $$k(\mathbf{x}, \mathbf{x}')>0$$
- it must be symmetric $$k(\mathbf{x}, \mathbf{x}')=k(\mathbf{x}', \mathbf{x})\,.$$

In general, we can decompose any kernel $$k(\mathbf{x}, \mathbf{x}')$$ as

$$k(\mathbf{x}, \mathbf{x}') = \tilde{k}(\mathbf{x}- \mathbf{x}') + \bar{k}(\mathbf{x}+ \mathbf{x}')$$

If we assume translational invariance
$$k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}+\mathbf{a}, \mathbf{x}'+\mathbf{a})$$
we must restrict ourself to $$k(\mathbf{x}, \mathbf{x}') = \tilde{k}(\mathbf{x}- \mathbf{x}')\,.$$
In this case the GP is said to be **stationary**.

