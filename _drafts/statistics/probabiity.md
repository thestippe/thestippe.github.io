---
layout: post
title: "A gentle reminder on probability"
categories: /statistics/
subcategory: Introduction
tags: /statistics_overview/
date: "2024-01-03"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Some notation and convention"
section: 0
---

In the following we will use $p(\cdot)$ to denote any probability density or probability mass function.
As an example, $$p(x)$$ denotes the probability density for $x\,,$ while $p(x, \theta)$ indicates the joint
probability for $x$ and $\theta\,.$
We will use latin letters to indicate observed (or at least observable) quantities, while we will use greek letters
for unobservable ones.
With $p(x | \theta)$ we indicate the distribution of $x$ conditional on $\theta\,,$
therefore Bayes theorem reads

$$
p(\theta | x) = \frac{p(x | \theta)}{p(x)} p(\theta)
$$

For special pdfs we will use special letters. As an example, the pdf for the normal distribution
will be written as
$
\mathcal{N}\,.
$

With this notation we will use $$\mathcal{N}(y | \mu, \sigma)$$
to indicate that $y$ is distributed according to a normal distribution with mean $\mu$
and variance $\sigma^2\,.$
Equivalently, we will write

$$
y \sim \mathcal{N}(\mu, \sigma)\,.
$$

In order to avoid confusion, we will sometimes write $P$ for a probability.
We will use $F$ for the cumulative distribution function, so

$$
F(x) = P(u\leq x) = \int_{-\infty}^x du p(u)\,.
$$

We will use $\mathbb{E}[\cdot]$ to denote expectation values, so


$$
\mathbb{E}_p[x] = \int_{-\infty}^\infty du u p(u)\,.
$$

In Bayesian statistics we mostly deal with the following quantities:
- The likelihood $$p(y \vert \theta)$$ which describes the probability for the observed variable given the parameters.
- The prior $p(\theta)$ which is the probability for the parameters before observing the data.
- The posterior $p(\theta \vert y)\,,$ which encodes the probability for the parameters after having observed the data
- The posterior predictive $p(\tilde{y} \vert y)$ namely the probability for new observations of the data
- The prior predictive $p(\tilde{y})$ the probability of observing $\tilde{y}$ before our measurment.