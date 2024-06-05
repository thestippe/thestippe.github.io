---
layout: post
title: "Predictive checks"
categories: /statistics/
subcategory: "Bayesian workflow"
tags: /predictive_checks/
date: "2024-01-22"
section: 2
# image: "/docs/assets/images/perception/eye.jpg"
description: "Verifying the predictions of your model"
---

In this post we will collect two different
phases of the Bayesian workflow, namely the
**prior predictive checks** and the
**posterior predictive checks**.

The first one is aimed to ensure that your priors
are compatible with the current domain knowledge,
and it doesn't require the comparison of the model
prediction with the true data.

The posterior predictive checks, instead, are focused
on clarifying if your fitted model can catch the relevant
aspects of the data.

While the two phases are in principle different,
they use similar methods, and for this reason
they are collected in a single post.

## Prior predictive checks

When you are performing a prior predictive check,
you are verifying if your model is flexible
enough to include what you know about the problem.

There are many ways you can perform this,
and this can be done by generating fake data
according to what you know about the problem
and then fit them.

If you know nothing, you may decide to pick a 
dataset sub-sample and ensure that it is included
within the prior predictive. Remember,
however, that you do not want to *fit* the sub-sample,
otherwise you may end up overfitting your data.

## The twitter data again

Let us consider again the dataset introduced
in the [post on the Negative Binomial](/statistics/negbin).
We already discussed some checks in that post,
and we carefully chose the value of the parameters
by making an educated guess on the order of magnitude
of the interactions.

Let us now assume that this time we made some
error in the procedure and we take

$$
\begin{align}
\theta & \sim \mathcal{B}(1/2, 1/2)\\
\nu & \sim \mathcal{Exp}(20)
\end{align}
$$

It is not rare to mess up with the parametrization,
so we may have confused $\lambda$ with its inverse
(it is more common than what you may imagine).
While the imported libraries and the data are the same,
this time the model reads as follows

```python
with pm.Model() as pp0:
    nu = pm.Exponential('nu', lam=50)
    theta = pm.Beta('theta', alpha=1/2, beta=1/2)
    y = pm.NegativeBinomial('y', p=theta, n=nu, observed=yobs)
```

Let us now sample the prior predictive

```python
with pp0:
    prior_pred_pp0 = pm.sample_prior_predictive()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(
prior_pred_pp0.prior_predictive['y'].values.reshape(-1), bins=np.arange(20), density=True)
ax.set_xlim([0, 20])
```

![The histogram of the prior predictive distribution](/docs/assets/images/statistics/predictive/prior_predictive.webp)

We can see that the value $y=0$ has a probability
greater than the $90\%.$
Let us now compute this probability.

```python
(prior_pred_pp0.prior_predictive['y'].values.reshape(
    -1)==0).astype(int).sum()/len(
    prior_pred_pp0.prior_predictive['y'].values.reshape(-1))
```

<div class='code'>
0.975
</div>

It doesn't really make much sense to start
from a model which predicts that the $97\%$
of our tweets have zero interaction.

At this point, a wise Bayesian would go back and
check again the model. Let us see what happens
to the unwise Bayesian who fits the model
despite the unreasonable conclusion which
follows from the prior.

```python
with pp0:
    trace_pp0 = pm.sample(random_seed=rng, chains=4, draws=5000, tune=2000)

az.plot_trace(trace_pp0)
```

![The trace plot of the unwise Bayesian](/docs/assets/images/statistics/predictive/trace-wrong.webp)

With the old model, the value of $\nu$
was peaked at $\nu = 2\,,$
while this time we have a peak at around $0.8\,.$
This means that our inference is strongly
biased by our prior.
In fact, our posterior predictive
is much worse that the one in the old post.

```python
with pp0:
    pp = pm.sample_posterior_predictive(trace)
az.plot_ppc(pp)
```

![The posterior predictive plot](/docs/assets/images/statistics/predictive/posterior_predictive.webp)

While our old model gave us,
on average, the correct probability for the tweet
with the lowest interaction number,
this time we overestimate the number
of tweets with few interactions.

Also in this case, the wise Bayesian would
stop and go back to the model construction.

## Conclusions

We discussed how to implement some prior predictive
and posterior predictive check,
together with the risks that comes by
not doing them.
