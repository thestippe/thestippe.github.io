---
layout: post
title: "Hypothesis testing"
categories: /statistics/
tags: /hypothesis_testing/
# image: "/docs/assets/images/perception/eye.jpg"
description: "Introduction to hypothesis testing"
publish: False
published: False
---

We now move from point estimate to hypothesis testing.
Many of the ideas that we introduced when we discussed the point
estimates clearly applies in this context, so we will not discuss
them again.

## Hypothesis testing

In hypothesis testing, the space $\Theta$ is decomposed
as $\Theta = \Theta_H \cup \Theta_K$ where $\Theta_H$
is the subspace where the hypothesis is true,
while $\Theta_K$ is the class of alternatives.

Let $d_0$ and $d_1$ be the decisions to accept or reject
the hypothesis respectively.
A testing procedure assigns to each $x$ one of these two decisions,
we therefore decompose the sample space into two subsets $S_0$ and $S_1\,,$
named the **acceptance region** and the **rejection region**.

When we perform a test, we may perform two kind of errors:
- we may reject $H$ when $H$ is true (type I error)
- we may accept $H$ when $H$ is false (type II error)

What we would like to perform is to keep both of them as small as possible,
but we cannot control both of them simultaneously.
What we can do is to fix a maximum allowed value for the probability
of incorrectly rejecting $H$ and look for the test which minimizes
the probability of incorrectly accepting $H$ within this space.

Mathematically, we require

$$ P_\theta( T(X) = d_1 ) \leq \alpha\,  \forall \theta \in \Theta_H $$

The quantity $\alpha$ defined above is the **level of statistical significance**
of the test.
We then maximize

$$
P_\theta(T(X) = d_1) \, \forall \theta \in \Theta_K
$$

The function, defined over $\Theta$

$$
1-\beta(\theta) = P_\theta(T(X) = d_1)
$$

is defined as the **power** of the test.
If there exists a test $T$ such that $1-\beta(\theta)$ is maximum
for any $\theta \in \Theta_K\,,$ we say it is the
**Uniformly Most Powerful** (UMP) test.

Providing the level of acceptance $\alpha$ may sometimes be not
enough, and it could be instructive to figure out
the smallest $\alpha$ compatible with the observed outcome $x\,,$
and this quantity is known as the **p-value**.


Sometimes working with a binary outcome may be too restrictive, as it may
not exist an optimal solution.
It is therefore convenient to allow for the decision to be **randomized**:
we define a function $0\leq \phi(x)\leq 1\,,$ and we set the outcome of
the test equal to $d_0$ with probability $\phi(x)$
while the outcome is equal to $d_1$ with probability $1-\phi(x)\,.$
It is clear that, if $\phi(x)$ is a function which only takes
values 0 and 1, we end up in the previous deterministic case.

## Confidence intervals

Consider the case of a one dimensional problem, where you want
to determine an lower bound $$\underline{\theta}$$ for $\theta$.
Of course, you cannot determine it with 100% probability,
since your upper bound depends on the data $X$, which is a random
variable.
What you can do is to fix a small $\alpha$ such that

$$P(\underline{\theta}(X) \leq \theta) \geq 1-\alpha\, \forall \theta$$

The function $\underline{\theta}(X)$ is named the **lower confidence
bound** for $$\theta\,.$$

The problem of determining an upper bound is clearly totally symmetric.

Finding a confidence interval for $\theta$ is clearly
a special case of hypothesis testing, so there is a **duality**
between confidence interval and hypothesis testing.

## Conclusions

We discussed few general concepts related to hypothesis
testing and confidence interval determination in frequentist
statistics.
Starting from the next post, we will start looking at how Bayesian statistics
deals with these issues.
