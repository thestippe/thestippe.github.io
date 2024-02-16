---
layout: post
title: "Common discrete probabilities"
categories: /statistics/
subgategory: Introduction
section: 3
tags: /discrete_probabilities/
date: "2024-01-08"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Derivation of some useful one-dimensional discrete probabilities"
---

In order to be able to build some model, it is important to 
know some elementary probability distribution, together
with the underlying process that generated it.
We will now discuss some of the most important discrete probability distributions.
We will also provide the lowest order moments,
as they can be really useful when trying to perform rough estimates.

## Bernoulli distribution

Let us consider an experiment which may have two possible outcomes.
The outcomes are traditionally named *success* and *fail*,
so 
$$\Omega = \left\{fail, success\right\}$$
For this kind of experiment it is convenient to define
the random variable 

$$X(fail) = 0, X(success) = 1$$

The most general pmf for this random variable is the **Bernoulli**
distribution

$$
p(x | p) = 
\begin{cases}
p & x = 1\\
q = 1-p & x = 0 \\
\end{cases}
$$

where our parameter $p$ may take any value in $[0, 1]\,.$

Notice that the general formula for $p(x | p)$ defines a family of distributions
rather than a distribution unless the parameter $p$ is fixed to a particular value.
It is customary however to name it as a distribution, and we will usually stick to this convention.

$$
\mathbb{E}[X] = \sum_{x=0, 1} x p(x | p) = p
$$

$$
Var[X] = \sum_{x=0, 1} (x-p)^2 p(x | p) = p^2(1-p) + (1-p)^2 p = p(1-p)
$$

## Binomial distribution

Let us now consider $n$ independent Bernoulli trials $X_1,\dots,X_n$,
the Binomial distribution describes the probability to
obtain $k$ successes out of the $n$ trials.
In order to derive it, we observe that, thanks to our choice of the Bernoulli
random variables $X_i\,,$ we can consider
$$ X = \sum_{i=1}^n X_i \,.$$

The probability that $X_1=1,\dots,X_k=1,X_{k+1}=0,\dots,X_n=0\,.$
is given by $p^k(1-p)^{n-k}\,.$
We are however not interested into the above probability,
but in the probability that *any* group of $k$ variables
takes value 1 while the remaining takes the value 0,
and this implies that we must multiply the above probability by the
number of possible groups of $k$ elements out of $n$ objects,
namely by $$C(n, k) = \binom{n}{k} = \frac{n!}{(n-k)!k!}\,.$$

$$
p(k | n, p) = \binom{n}{k} p^k(1-p)^{n-k}
$$

where $p \in [0, 1]\,,$ $n=1,2,\dots$ and $k=0,\dots,n$

Since the Binomial distribution is the sum of $n$ independent Bernoulli distribution,
it is straightforward to get

$$
\mathbb{E}[X] = n \mathbb{E}[X_1] = n p
$$

and

$$
Var[X] = n Var[X_1] = n p (1-p)
$$

## Negative binomial distribution

The negative binomial distribution gives the probability
to observe $k$ failures before observing $r$ (fixed) successes.
By assumption, the $k+r$-th event is a success, and it happens with probability $p$.
As before we can show that the probability to observe
$k$ failures out of $r+k-1$ events is given by $\binom{r+k-1}{k}(1-p)^{k}p^{r-1}$
so

$$
p(k | r, p) = \binom{r+k-1}{k} (1-p)^k p^r
$$

In this case we have $p\in [0, 1]\,,$ $r=1,2,\dots$ and $k=0,1,\dots\,.$

The negative binomial distribution with $r=1$ is named the **geometric**
distribution.
The geometric distribution turns out to be very useful,
as you can think about the general negative binomial distribution as a sum of r
geometric distribution.
If $Y_1$ is the waiting time for the first success, $Y_2$ the one for the second success,\dots,
$Y_r$ the waiting time for the $r$-th success, then you can consider the total waiting
time
$$ X = Y_1 + Y_2 + \dots + Y_r\,.$$

We have that

$$
\begin{align}
\mathbb{E}[Y_1] &= \sum_{k=0}^\infty k p (1-p)^k = (1-q) \sum_{k=0}^\infty k q^k\\
&
= (1-q) q \sum_{k=0}^\infty k q^{k-1}
\\ &
= (1-q) q \partial_q \sum_{k=0}^\infty q^k
\\ &
= (1-q) q \partial_q (1-q)^{-1} 
\\ &
= (1-q) q (1-q)^{-2}
\\ &
= \frac{q}{1-q} 
\\ &
= \frac{1-p}{p}
\end{align}
$$

In the same way we get

$$
Var[Y_1] = \frac{1-p}{p^2}
$$

So

$$
\mathbb{E}[X] = r \frac{1-p}{p}
$$

and

$$
Var[X] = r\frac{(1-p)}{p^2}
$$

## Poisson distribution

Let us consider an experiment,
and let us assume that, in a time $\delta t$, on average,
 $\mu $ events happen.

We assume that each event is independent on the others,
the probability that the site gets $k$ events
must be given by:

$$P(X=k) \propto \frac{\mu^k}{k!}$$

where the denominator has been introduce
since we don't care the order of the events.
We can get the overall normalization constant by normalizing the probability to one

$$
C \sum_{k=0}^\infty \frac{\mu^k}{k!} = C e^{\mu}= 1 $$

so

$$
p(k | \mu) =e^{-\mu } \frac{ \mu ^k}{k!}$$

with $\mu > 0$ and $k=0,1,\dots\,.$

We have

$$
\begin{align}
\mathbb{E}[X] & = 
e^{-\mu} \sum_{k=0}^\infty k \frac{\mu^k}{k!}
\\ &
=e^{-\mu} \sum_{k=1}^\infty k \frac{\mu^k}{k!}
\\ &
=e^{-\mu} \sum_{k=1}^\infty \frac{\mu^k}{(k-1)!}
\\ &
=\mu e^{-\mu} \sum_{k=1}^\infty \frac{\mu^{k-1}}{(k-1)!}
\\ &
=\mu e^{-\mu} \sum_{k=0}^\infty \frac{\mu^{k}}{k!}
\\ &
= \mu
\end{align}
$$

Analogously we can obtain

$$
Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2 = \mu
$$

## Discrete uniform

The discrete uniform distribution is the one which assigns equal
probability to $n$ possible outcomes.
If we define the random variable $X=1,\dots,n\,,$ we immediately have

$$
p(x) = \frac{1}{n}
$$

Also in this case the computation of the mean is quite straightforward

$$
\mathbb{E}[X] = \sum_{k=1}^n \frac{k}{n} =\frac{1}{n} \frac{n (n+1)}{2} =\frac{n+1}{2}
$$

The computation of the variance is a little bit more tedious, but at the end the result is

$$
Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2 = \frac{(n+1)(2n+1)}{6} - (\frac{n+1}{2})^2 = \frac{n^2-1}{12}
$$

## Categorical distribution

The categorical distribution is the most general distribution
of $n$ events.
Given the random variable $X=0,\,\dots,n-1\,,$ 
and a vector $$(p_0,\dots,p_{n-1})$$ with $p_i \in [0, 1]$ and $\sum_{i=0}^{n-1} p_i = 1$
we assign

$$
p(x=i | p_0,\dots,p_{n-1}) = p_i
$$

The categorical distribution corresponds to the Bernoulli distribution
for $n=2\,,$ while if $p_0=p_1=\dots=p_{n-1}=\frac{1}{n}$ it reduces
to the discrete uniform distribution.

In this case there is no simple formula for the expected values.

## Conclusions
We have discussed some of the most common discrete distributions.
In the next post we will discuss some relevant continuous distribution.
