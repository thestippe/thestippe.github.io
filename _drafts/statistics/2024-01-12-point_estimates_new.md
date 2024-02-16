---
layout: post
title: "Point estimates"
categories: /statistics/
tags: /point_estimates/
# image: "/docs/assets/images/perception/eye.jpg"
description: "Introduction to the theory of point estimation"
publish: False
published: False
---

In the last post we discussed about decision theory,
in this post we will discuss some important aspects of
point estimation in the frequentist perspective.

## Point estimation

In the point estimate we are looking for a $T(X)$ such that

$$
MSE[T] = 
        \max_{\theta \in \Theta}\mathbb{E}_\theta[(T(X)-g(\theta))^2]
$$

is minimum.
Notice that

$$
\begin{align}
MSE[T] & = 
\mathbb{E}_\theta[(T(X)-g(\theta))^2]
=
\mathbb{E}_\theta[((T(X)-\mathbb{E}_\theta[T(X)])-(g(\theta)-\mathbb{E}_\theta[T(X)]))^2]
\\
&
=
\mathbb{E}_\theta[(T(X)-\mathbb{E}_\theta[T(X)])^2]+\mathbb{E}_\theta[(g(\theta)-\mathbb{E}_\theta[T(X)])^2]
-2\mathbb{E}_\theta[(T(X)-\mathbb{E}_\theta[T(X)])(g(\theta)-\mathbb{E}_\theta[T(X)])
]
\\
&
=
\mathbb{E}_\theta[(T(X)-\mathbb{E}_\theta[T(X)])^2]+(g(\theta)-\mathbb{E}_\theta[T(X)])^2
\\
&
= Var[T(X)] + Bias[T(X), g(\theta)]^2
\end{align}
$$

where the **bias** of an estimator is defined as

$$\mathbb{E}_\theta[T(X)-g(\theta)]\,.$$

If the bias is 0 for any $\theta\in\Theta$ we say that $T$ is an **unbiased estimator** for $g(\theta)\,.$

It is generally not guaranteed that an unbiased estimator for $g(\theta)$
exists. If it exists, then $g(\theta)$ is said to be **U-estimable**.

The remaining term, the estimator variance, does not
depend on the estimand, but it's an intrinsic property of the estimator.
From the above relation it should be clear that, given two unbiased
estimators for $$g(\theta)\,,$$ we should choose the one with the smallest
variance.

We say that $T(X)$ is the **Uniformly Minimum Variance Unbiased Estimator** (UMVUE)
if $$Var[T(X)] < Var[U(X)]\, \forall \theta \in \Theta, \forall U(X)$$ unbiased
estimator of $g(\theta)\,.$

If we find a UMVUE for $g(\theta)\,,$ then it minimizes the quadratic risk
function, to it can be consider the optimal point estimator for $g(\theta)$
according to our criteria.

## Consistency

Sometimes it is not possible to find an UMVUE for $g(\theta)\,,$
and in these cases one has to find a balance the two terms
$Var[T(X)]$ and $Bias[T(X), g(\theta)]\,.$

One possibility is to select an estimator which becomes more and more precise
as the sample size grows.
In mathematical terms, the concept "becomes more and more precise"
requires to be specified more precisely.
We say that an estimator is **consistent** if, 
$$\forall \varepsilon > 0$$

$$
\lim_{n \rightarrow \infty} P(|T(X_n) - g(\theta)|<\varepsilon) = 1\,  \forall \theta \in \Theta
$$

Notice that consistency does not imply unbiasedness, as there may be a bias
for finite $n$ but, in the limit, the bias may disappear.
The most known case for this is the naive estimator for the variance.

$$
\hat{S}_n = \frac{1}{n}\sum_{i=1}^n \left(X_i-\bar{X}\right)^2
$$

In fact, let us now consider a sample of independent random variables
with mean $\mu$ and variance $\sigma\,.$
We can choose $\mu=0$, as this doesn't affect the result,
since we can always define $Y_i = X_i - \mu$ and $\bar{Y} = \bar{X}-\mu$ and leave $S_n$
unchanged.

$$
\begin{align}
\mathbb{E}[\hat{S}_n] & =
\mathbb{E}\left[ \frac{1}{n}\sum_i (X_i - \bar{X})^2 \right]
=
\frac{1}{n}\mathbb{E}\left[ \sum_i (X_i^2 + \bar{X}^2 -2 X_i \bar{X}) \right]
=
\sigma^2 
+ \mathbb{E}\left[ \bar{X}^2 \right]
-2 \sum_i \frac{1}{n}\mathbb{E}\left[  X_i \bar{X} \right]
=
\sigma^2 
+ \frac{\sigma^2}{n}
-\frac{2}{n} \sum_i \mathbb{E}\left[  X_i^2 \right]
\\
&
= \sigma^2 \left(1-\frac{1}{n}\right) = \sigma^2 \left(\frac{n-1}{n}\right)
\end{align}
$$

Above we used
$$\mathbb{E}[X_i^2] = Var[X_i] = \sigma^2\,,$$
$$\mathbb{E}[\bar{X}^2] = Var[\bar{X}^2] = \frac{\sigma^2}{n}$$
and
$$\mathbb{E}[X_i \bar{X}] = \mathbb{E}[X_i^2] = Var[X_i] = \sigma^2\,.$$

The unbiased (and consistent) version of $\hat{S}_n$ is

$$
S_{n} = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2
$$

Also the reverse holds: unbiasedness does not imply consistency.
Given an iid set of Gaussian random variables $X=(X_1,\dots,X_n)$
with mean $\mu$ and variance $\sigma\,,$
the estimator $T(X)=X_1$ is an unbiased estimator for $\mu$, as its expectation value
is $\mu\,,$ but it's not consistent, as its variance is always
equal to $\sigma\,.$

## Sufficient statistics

The task of finding a $T$ for each $g$ seems a hard task. 
There are some cases, however, where it is sufficient to find
a set of $T$ for $\theta$ to have enough informations
to solve our minimization problem for any $g$,
and these are the families which admit a sufficient
complete set of statistics.

A statistic is said to be **sufficient** for a parameter $\theta$
if the joint probability distribution of the data $X$ is conditionally independent of the parameter $X$ given the value of the sufficient statistic $T(X)$ for the parameter.
In other words, for a sufficient statistic we are allowed to replace
$\theta$ with $T(X)\,.$

A statistic is said to be **complete** if

$$
\mathbb{E}_\theta[f(T(X))] = 0
$$

implies $$f(t) = 0$$ almost everywhere with respect to $\mathcal{P}\,.$

If a statistic is complete, then no other information can be added to
the model once $T(X)$ is given.

The [Lehmann-Shaffé theorem](https://en.wikipedia.org/wiki/Lehmann%E2%80%93Scheff%C3%A9_theorem) states that,
if $T$ is a complete sufficient statistic for $\theta$ and
$$\mathbb{E}_\theta[f(T(X))] = g(\theta)\,,$$
then $$f(T(X))$$ is the UMVUE for $g(\theta)\,.$

A very important class of families is the **exponential families**,
which are families where the pdf takes the form

$$ p(x | \theta) = h(x)e^{\eta(\theta) \cdot T(x) - A(\theta)}$$

and in this case $T$ is a sufficient statistic.
Notice that the statistic for an exponential family may not be complete.
We say that a sufficient statistic is **minimal** if we can
express any other statistic it terms of it, so if any other statistic
$$S$$ there exist a measurable function $$f$$ such that
$$S = f(T)\,.$$

An important result is the [Pitman-Koopman-Darmois](http://yaroslavvb.com/papers/koopman-on.pdf) lemma, which tells us that *only* exponential families
admit sufficient statistics for an iid sample.

Most of the distributions that we discussed define an exponential family distribution, but not all of them.
In particular, you can verify that the Gaussian, the multinomial, the negative binomial, the Gamma and the Beta distributions
define an exponential family distribution (together with their special cases, the binomial and the $\chi^2$ distribution). 

## Maximum Likelihood Estimator

Exponential families may allow you to find a complete sufficient statistic, however it is not
always the case.
If your task is to find a suitable estimate for $\theta\,,$ than instead on relying on 
decision theory you may leverage another principle, namely the **likelihood principle**.

The likelihood of the model $p(x | \theta)$ is the pdf/pmf of the model where the random variables
are fixed to the observed values, and it is a function of the parameter $\theta\,.$

<div class='emphbox'>
Given a statistical model, all the experimental evidence in a sample regarding its parameters is
encoded in the likelihood function.
</div>

Our aim is to find the value of $\theta$ that makes the observed data the most probable as possible
within the family of distributions.
So we define the **Maximum Likelihood Estimator** for $\theta$
as

$$
\theta_{ML} = \arg \max_{\theta \in \Theta} p(x | \theta)
$$

This estimator is very common, as it simply requires to maximize a function rather than
to minimize the entire risk.
This may be a suboptimal choice, as it doesn't minimize the risk,
but it may make life easier.

## Conclusions
We discussed some strategies to solve the point estimate problem
in the frequentist framework. In the next post we will discuss about testing.
