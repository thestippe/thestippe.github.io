---
layout: post
title: "Some notation about probability"
categories: /statistics/
subcategory: Introduction
tags: /probability/
date: "2023-11-28"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Notation and conventions"
section: 0
---

In the following we will use $$p$$ do refer to any probability density function
or probability mass function.
We will use latin letters to refer to observed (or observable) quantities,
while we will use greek letters to refer to unobservable ones.

If $x$ and $y$ are two random variables, we will use $$p(x, y)$$ for the joint
distribution function, while $$p(x \vert y)$$ will be the conditional distribution function.
The two quantities are related by

$$
p(x \vert y) = \frac{p(x, y)}{p(y)}
$$

where 

$$p(y) = \int dx p(x, y) \,.$$

Similarly, we have

$$
p(y \vert x) = \frac{p(x, y)}{p(x)}
$$

where 

$$
p(x) = \int dy p(x, y)
$$

so we immediately have

$$
p(x, y) = p( x \vert y) p(y) = p(y \vert x) p(x)
$$

or, assuming $$p(x) \neq 0$$

$$
p(y \vert x) = \frac{p( x \vert y) p(y)}{p(x)}
$$

which is nothing but Bayes theorem.
Therefore, when building our model, we will assume that our data $y$ is distributed according
to some probability distribution function
$$p(y \vert \theta)$$ which is usually called the **likelihood function**
having $$\theta$$ as parameter. We will use an equivalent notation

$$
y \vert \theta \sim p \,.
$$

We will usually refer to the probability distribution function family by using its name or its
symbol. As an example, if we assume that $y$ is distributed according to a gaussian distribution
with mean $\mu$ and variance $\sigma\,,$ we will write

$$
y \sim \mathcal{N}(\mu, \sigma)\,.
$$

In order not to overwhelm the reader with the notation, we will introduce the symbols indicating
the different probability distribution functions
as soon as they are needed.

The quantity $$p(\theta)$$ is usually named the **prior distribution function** for the parameters.
The quantity $$p(\theta \vert y)$$ is the **posterior distribution function**,
which is related to the likelihood and to the prior by

$$
p(\theta \vert y) = \frac{ p(y \vert \theta) p(\theta)}{p(y)}\,.
$$

Since $p(y)>0$ does not depend on theta, we can rewrite the above as

$$
p(\theta \vert y) \propto p(y \vert \theta) p(\theta)\,,
$$

where as usual the denominator $$p(y)$$ can be determined by normalizing the posterior to 1

$$
p(y) = \int d\theta p(y \vert \theta) p(\theta)\,.
$$

We can also determine the probability distribution for unobserved data 

$$
p(\tilde{y} \vert y) = \int d\theta p(\tilde{y} \vert \theta) p(\theta \vert y) d\theta
$$

and $$
p(\tilde{y} \vert y)$$ is named the **posterior predictive distribution**.

We will use $P$ to indicate the probability. Therefore, if $\theta$ is a continuous
variable and $\theta_0$ some fixed number

$$
P(\theta < \theta_0) = \int_{-\infty}^{\theta_0} p(\theta) d\theta \,.
$$

The above quantity is also the **cumulative distribution function** evaluated at $\theta_0$,
also indicated as $F(\theta_0)\,.$

The **expected value** for $\theta$ is

$$
\mathbb{E}[\theta] = \int \theta p(\theta) d\theta\,.
$$

We will also often use the **variance** 

$$
Var[\theta] = \mathbb{E}[(\theta - \mathbb{E}[\theta])^2] = \mathbb{E}[\theta^2] - \mathbb{E}[\theta]^2\,.
$$

The **covariance** between two variables $$\theta$$ and $$\phi$$ is

$$
Cov[\theta, \phi] = \mathbb{E}[(\theta - \mathbb{E}[\theta])(\phi - \mathbb{E}[\phi])] 
= \mathbb{E}[\theta \phi] - \mathbb{E}[\theta]\mathbb{E}[\phi]\,.
$$

The **correlation** between two variables is defined as

$$
Corr[\theta, \phi] = \frac{Cov[\theta, \phi]}{\sqrt{Var[\theta] Var[\phi]}}
$$
