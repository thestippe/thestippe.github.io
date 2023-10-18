---
layout: page
title: "The frequentist perspective"
categories: course/appendices/
tags: /estimation/
---

While most of this blog is about Bayesian statistics, in this post
we will try and give an overview to some of the most relevant concepts
about frequentist statistics.

We generally assume that we are trying and make some statements about the properties
of a population $\mathbb{P}$.
In **parametric inferential statistics** we assume that our population
is distributed according to some family of distributions.
As an example, we could assume that our population is distributed according to
a normal distribution with unknown mean and known variance, and we would like to determine
if the mean of the distribution is somehow compatible with zero.

There are many ways to build those family of distributions, and one of the
most useful is the **exponential family form**.
For this kind of families we assume that the probability distribution
function takes the form

$$
p(x \vert \theta) = h(x) g(\theta)e^{T(x)\eta(\theta)}\,.
$$


We define a **sufficient statistic** a statistic (which is nothing
but a quantity that can be computed from the data) such that no other statistic
can provide additional informations about our distribution.
As an example, in our previous example, a sufficient statistics
for the mean of the distribution $\mu$ is the arithmetic mean of the population:

$$
T(x) = \frac{1}{N}\sum_{j=1}^N x_i\,.
$$

A necessary and sufficient condition for a distribution family to admit a sufficient statistics is to be an exponential family distribution,
and in this case the sufficient statistics is $T(x)$.

In this context, $\eta$ is called the **natural parameter** of the distribution family,
while one refers to $e^{A(\eta)}$ as the partition function.

In the jargon one often refers to the distribution family as the distribution
itself. This is not very precise, as the distribution is an element
of the distribution family, but it is commonly accepted.

In most case we won't deal with the entire population, but only with a (possibly random)
sub-sample of it $$\{X_1,...,X_n\}$$.
In this case we cannot exactly calculate the parameter of interest, but we can only
**estimate** it, and this is why we refer to the parameter as the **estimand**.
We call an **estimator** a map from the sample space to the space of the estimates.

In our usual example, the arithmetic mean of the sample is an estimator of the 
parameter $\mu$.

Given an estimator $$\hat{\theta}$$ of a parameter $\theta$ we define its bias as

$$
E[\hat{\theta}-\theta]
$$

We say that an estimator is unbiased it the bias is zero.

Let us consider a sample of $n$ iid normally distributed observations
with mean $\mu$ and variance $\sigma^2$.

The arithmetic mean of the sample,
defined as
$$
\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i
$$
is an unbiased estimator of the population mean $\mu\,,$ w

On the other hand, the uncorrected estimator for the sample variance,
defined as

$$
S^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

is a biased estimator, while the unbiased estimator is $\frac{n}{n-1}S^2$.

There are two kind of estimates: **point estimates** and **interval estimates**.
In the case of point estimate, the estimate space has the same dimension
of the parameter space, as the estimate provides a point in the parameter
space.
The main issue of point estimates is that they provide no information
about the uncertainty that is associated with the estimate itself.

Confidence interval/region estimates, on the other hand, provide an interval/region
(depending if we are working with a one dimensional parameter space or with a
higher dimensional space).
More precisely, a confidence interval for $\theta$ with confidence level
$\gamma = 1-\alpha$ is an interval $(u(X), v(X))$ such that

$$P(u(X)\leq \theta \leq v(X)) = \gamma $$

We should always keep in mind that our parameter $\theta$ is given,
and our confidence interval does not tell us anything about the probability
for $\theta$ of being inside the interval.
All we can say is that, if we repeat an experiment many times and each time
we compute the confidence interval for the sample, a fraction
of times $\gamma$ our confidence interval will contain the true parameter $\theta$.

