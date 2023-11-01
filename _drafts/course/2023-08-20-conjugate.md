---
layout: post
title: "Conjugate models"
categories: course/intro/
tags: /conjugate/
description: "If you want to tackle your problem analytically"
image: "/docs/assets/images/conjugate/Bayes_icon.png"
---

We previously mentioned the concept of conjugate models, in this
post we will have a deeper look at this kind of model.

## The Beta-Binomial model as conjugate model

Let us consider again the binomial model with uniform prior:

$$
\begin{align}
&
y \sim Binomial(\theta, n) \\
&
\theta \sim Uniform(0, 1)
\end{align}
$$

so

$$
p(y | \theta) \propto \theta^y (1-\theta)^{n-y}
$$

and, since the prior does not depend on $\theta\,,$ we have that

$$
p(\theta) \propto 1
$$

by using Bayes theorem we have that

$$
p(\theta | y) \propto p(y | \theta) p(\theta) \propto \theta^y (1-\theta)^{n-y}
$$

We can now consider a more general family of prior distribution,
namely the Beta distribution:

$$
p(\theta | \alpha, \beta) \propto \theta^{\alpha-1} (1-\theta)^{\beta-1}\,.
$$

If we take this as a prior and we use the Bayes theorem

$$
p(\theta | y, \alpha, \beta) \propto p(y |\theta, \alpha, \beta) p(\theta | \alpha, \beta)
\propto \theta^{y} (1-\theta)^{n-y} \theta^{\alpha-1} (1-\theta)^{\beta-1}
\propto \theta^{\alpha+y-1} (1-\theta)^{\beta+n-y-1}
$$

From the last formula we see that the posterior distribution has the same
form of the prior distribution. In this case we say that the
Beta distribution is a conjugate prior of the Binomial distribution.

By normalizing the distribution to one, we get

$$
p(\theta | y, \alpha, \beta) = \frac{1}{B(\alpha+y, \beta+n-y)}
\theta^{\alpha+y-1} (1-\theta)^{\beta+n-y-1}
$$

or, equivalently,

$$
\theta | y, \alpha, \beta \sim Beta(\alpha+y, \beta+n-y)\,.
$$

Up to the middle of the last century, conjugate models were widely
used in Bayesian statistics, as it was the only kind of analytically solvable
model.
However, nowadays, one can build and study any kind of model, thanks
to the MCMC sampling techniques.
It is however very useful to have some knowledge about conjugate models.

When you develop complex models, the assessment of a prior distribution
can be a tough task.
Conjugate models will allow you to make an educated guess on your prior,
since it is very easy to understand how does the data affect the posterior.

## An application of conjugate models

As an example, let us assume your friend who likes fooling
people.
Your friend has a coin, and he claims it is a fair coin within very high approximation.
He suggests to toss the coin 10 times to see if the coin is fair.
You decide that you will use a beta-binomial model to analyze the data.
You don't know if the coin is more likely to give head or tail, so your
conjugate model will have $\alpha = \beta\,.$

You decide to choose $\alpha$ in such a way that, if you get 5 times head
and 5 times tail, then the $80\%$ of the posterior should lay between 0.25 and 0.75.

Your friend tells you that it really looks a too stringent constraint, and h

We can then find $\alpha$ by using the following graphics:

```python
from matplotlib import pyplot as plt
from scipy.stats import beta
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(-5, 10, 0.02)
ax.plot(x, beta(x, x).cdf(0.25), color='r')
ax.plot(x, beta(x, x).cdf(0.75), color='b')
ax.axhline(y=0.1, color='k')
ax.axhline(y=0.9, color='k')
```

![Alt text](/docs/assets/images/conjugate/conjugate.jpg)

The point where the cumulative distribution functions
evaluated at $0.25$ ($0.75$) passes through $0.1$ ($0.9$),
represents $\alpha-5\,.$
This happens approximately at $x=3$, so $\alpha=-2$ is a good prior
(we don't need to _precisely_ choose alpha, a graphical method is thus sufficient).
You should keep in mind that this prior is not a proper
prior, as it does not integrate to 1.
This is not a problem, but you should be careful in
using it and always underline this when writing
your reports.

If you want to have a list of the most common conjugate models, take
a look at [this](https://en.wikipedia.org/wiki/Conjugate_prior) Wikipedia page,
while an exhaustive discussion about this kind of models can be fount
in Andrew Gelman's textbook (see the [resources](/links/) page).

## Conclusions and take-home message
- Conjugate models allow you to have an analytical way to link your parameters to observable quantities
- You can easily formulate a constraint on your priors in terms of effects on the posteriors (I choose my prior in such a way that, if I see this outcome I want my posterior to behave this way).
- Priors chosen in terms of effects on the posterior are very easy to understand, criticize and, if necessary, improve.
