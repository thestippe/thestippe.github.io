---
layout: post
title: "Introduction to Bayesian statistics"
categories: /statistics/
tags: /bayes_intro/
date: "2024-01-14"
section: 7
# image: "/docs/assets/images/perception/eye.jpg"
description: "Introduction to Bayesian statistics"
---

In the previous posts we discussed the theory
of point estimates, CI estimates and
hypothesis testing.
In this post we will analyze some of the major problems of frequentist
statistics, and what can Bayesian statistics do for us.

## Some issue with frequentist statistics

A point estimate for a parameter doesn't provide much information
about it, as there is no indication about the level of uncertainty
of the estimate.
We should therefore provide a confidence interval for it, but if
our model is too complicated, finding a good estimator could be a hard task.
If our sample size is large enough, we can always use the CLT to provide an estimate,
but if we are working with a small sample things may get really complicated.

There are also complications related to the correct interpretation of the confidence
interval, as often researchers fail to correctly interpret it.

In fact, a very common interpretation of a CI estimate for a parameter $\theta$
with significance level $\alpha$ is that the probability that $\theta$ is within the CI
is $1-\alpha\,.$
This of course is wrong, as in the frequentist interpretation $\theta$ is not
a random variable, so there is no probability associated to it.

As we will discuss in the future, there are also methodological issues in the correct implementation
of the frequentist statistics, and a very common error is the optional stopping.
What sometimes happens is that one uses a sample of size $n$
to test a hypothesis with significance $\alpha\,.$
If the hypothesis is rejected, then the result can be published, otherwise
 the experiment is repeated with an enlarged dataset of size $n_1$ and
the same $\alpha\,.$
However, with this procedure, in the long run you end up with a probability
of rejecting the null hypothesis equal to 1 even when the null hypothesis is true.

## Switching to the Bayesian framework

Under some circumstance you may consider to stick to the Bayesian framework rather
than inappropriately using frequentist statistics.

In Bayesian statistics you should first of all specify
a likelihood $p(x | \theta)\,,$ which encodes how data have been generated in your model given the parameters.
You should moreover provide a prior for your parameters $$p(\theta)\,,$$ which represents your beliefs about the parameter
before the experiment is performed.
You can then use Bayes' theorem

$$
p(\theta | x) \propto p(x | \theta) p(\theta)
$$

to encode the probability distribution of the parameters once the experiment has been performed.
You can moreover get the probability distribution for unobserved data $\underline{x}$ as

$$
p(\underline{x} | x) = \int d\theta p(\underline{x} | \theta) p(\theta | x)
$$

## How to compute the posterior

The main issue with Bayesian statistics up to some decade ago was that the analytic solution for the normalized $p(\theta | x)$
was only available for distributions belonging to an exponential family, and only for a limited set of priors, namely the **conjugate priors**.
Nowadays, however, thanks to a set of sampling techniques which goes under the name of **Markov Chains Monte Carlo** (MCMC),
it is possible to obtain a sample $\theta_1,\dots,\theta_K$ of the distribution $p(\theta | x)$ for virtually any likelihood
and any prior.
One can then obtain any expected value as

$$
\mathbb{E}[f(\theta)] \approx \frac{1}{K} \sum_{i=1}^K f(\theta_i)
$$

and the above approximation becomes exact in the limit $K \rightarrow \infty\,.$
Nowadays you MCMC has been implemented in more than one programming language,
and thanks to **Probabilistic Programming Languages** like PyMC or Stan you don't even have to specify by hand the priors or the likelihood,
you simply have to specify which is the distribution of your variable.

You can then provide the full posterior distribution for your variables, without any need to decide whether you should
report a point estimate or a confidence interval.
You can finally numerically minimize your risk function.

How to do all of this and much more using Python will be the major topic of the future posts of this section of the blog.


## Conclusions

We discussed some major issues with frequentist statistics, and we saw how to overcome these difficulties
by using Bayesian statistics.
In the next post we will see how to implement a Bayesian model in Python.
