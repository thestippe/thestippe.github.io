---
layout: post
title: "Model comparison"
categories: course/intro/
tags: /model-comparison/
image: "/docs/assets/images/model_comparison/loo.jpg"
description: "Understanding good and bad aspects of your models"
---

In the [last](/predictive-checks/) post we looked at how one can assess a model's ability to reproduce the data.
In this post we will look at a related topic, which is how we can compare two or more Bayesian models.
In fact, you rarely know from the beginning what is the most appropriate model to fit your data.
Most of the times you will find yourself building different models for the same dataset,
and a crucial part of your work will be to compare them.
Comparing model sometimes may be understood as choosing the best model,
but in most cases it means to asses which model is better to describe or predict some particular aspect of your data.
Model comparison can be done analytically in some case,
but most of the time it will be done numerically or graphically, and here we will give an overview of the most important tools.

```python
import numpy as np
from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
from scipy.stats import beta

plt.style.use("seaborn-v0_8-darkgrid")
```

## Bayes factors

Let us consider again our first example, where we had a sample of 79 yeast cells and we counted 70 alive cells and 9 death cells.
let us assume that we have two candidate models to describe our data:
model 1 has uniform prior, which mean that the prior is a beta distribution with $a=1$ and $b=1\,,$
while the second one has $a=b=10\,.$

```python
n = 79
y = 70

model_1 = {'a': 1, 'b': 1}
model_2 = {'a': 10, 'b': 10}

x_pl = np.arange(0, 1, 0.01)
fig = plt.figure()
ax = fig.add_subplot(111)
for k, model in enumerate([model_1, model_2]):
    ax.plot(x_pl, beta(a=model['a'], b=model['b']).pdf(x_pl), label=f"model {k+1}")
legend = plt.legend()
ax.set_ylabel(r"$\theta$")
ax.set_ylabel(r"$p(\theta)$  ", rotation=0)
fig.tight_layout()
```

![Priors](/docs/assets/images/model_comparison/priors.jpg)

Given the two models $M_1$ and $M_2$ we may ask which one we prefer, given the data. The probability of the model given the data is given by

$$ p(M_k | y) = \frac{p(y | M_k)}{p(y)} p(M_k) $$

where the quantity $p(y | M_k)$ is the **marginal likelihood** of the model. If we assign the same prior probability $p(M_k)$ to each model then we can simply replace $p(M_k | y)$ with the
marginal likelihood.

As usual, an analytic calculation is only possible in a very limited number of models.

One may think to compute $p(M_k| y)$ by starting from $p(y | \theta, M_k)$ and integrating out $\theta$ but doing this naively is generally not a good idea, as
this method is unstable and prone to numerical errors.

However can use the Sequential Monte Carlo to compare the two models, since it allows to estimate the (log) marginal likelihood of the model.

```python
models = []
traces = []

for m in [model_1, model_2]:
    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=m['a'], beta=m['b'])
        yl = pm.Binomial("yl", p=theta, n=n, observed=y)
        trace = pm.sample_smc(1000, return_inferencedata=True, random_seed=np.random.default_rng(42))
        models.append(model)
        traces.append(trace)
```

```python
az.plot_trace(traces[0])
```

![First trace](/docs/assets/images/model_comparison/trace_0.jpg)

```python
az.plot_trace(traces[1])
```

![Second trace](/docs/assets/images/model_comparison/trace_1.jpg)

What one usually computes is the **Bayes factor** of the models, which is the ratio between the posterior probability of the model (which in this case is simply the
ratio between the marginal likelihoods).

| $BF = p(M_1 \vert y)/p(M_2\vert y)$ | interpretation |
|-----------------|---------------|
| $BF<10^{0}$ | support to $M_2$ (see reciprocal) |
| $10^{0}\leq BF<10^{1/2}$ | Barely worth mentioning support to $M_1$ |
| $10^{1/2}\leq BF<10^2$ | Substantial support to $M_1$ |
| $10^{2} \leq BF<10^{3/2}$ | Strong support to $M_1$|
| $10^{3/2} \leq BF<10^2$ | Very strong support to $M_1$|
| $\geq 10^2$ | Decisive support to $M_1$|

We can now compute the Bayes factor as follows

```python
BF_smc = np.exp(np.mean(traces[0].sample_stats.log_marginal_likelihood[:, -1].values) - np.mean(traces[1].sample_stats.log_marginal_likelihood[:, -1].values))
np.log10(BF_smc)
```

> 2.0487805236

The Bayes factor is above 100, so we have a strong support for model 0.

We can better understand this result if we compare our estimate with the frequentist one, recalling that the confidence interval was $[0.81, 0.96]$

```python
az.plot_forest(traces, figsize=(10, 5), rope=[0.81, 0.96])
```

![Forest plot](/docs/assets/images/model_comparison/forest.jpg)

As we can see, our first model gives an estimate which is compatible
with the frequentist one, while the second HDI is not compatible
with the frequentist estimate.
We also have that the posterior predictive distribution of the first model is much
closer to the observed data than the one of the second model:

```python
ppc = []
for k, m in enumerate(models):
    with m:
        ppc.append(pm.sample_posterior_predictive(traces[k]))
```

```python
ppc[0].posterior_predictive['yl'].mean(['chain', 'draw']).values
```
> array(69.15525)


```python
ppc[1].posterior_predictive['yl'].mean(['chain', 'draw']).values
```
> array(63.75)

The first model predicts 69 alive cells, while the second one predicts 63.
So the first one is much closer to the observed number, which is 70.

## Leave One Out cross-validation

In the past, Bayes factor analysis was the most common method to perform
model selection.
However, according to many modern Bayesian statisticians, 
it should not be used for this purpose [^1].
The main criticism to this method is that you are both using
your data to fit the data and to check your model.
A better alternative is provided by the Leave One Out (LOO)
cross-validation.
LOO cross validation consists into using some metrics to
assess the probability of a datum where that datum is not uses
to fit the model.
There are many metrics that can be used,
and the most common ones are Aikane Information Criteria, Bayesian Information Criteria
(AIC and BIC respectively).
They are respectively given, for a model with $k$ parameters fitted by using $n$
points, as

$$
AIC = 2k - 2 \log \hat{L}
$$

$$
BIC = k\log(n) - 2 \log\hat{L}
$$

where $\hat{L}$ is the maximized value of the likelihood function.
However, none of them is truly Bayesian, as they are defined
using the maximum value of the likelihood function, while a more consistent
approach would use the average of the likelihood function [^2].
Arviz uses the Pareto Smoothed Importance Sampling (PSIS)
to estimate the LOO-Watanabe Aikane Information Criteria (WAIC), which is
the Bayesian version of the AIC.

Let us go back to the hurricanes dataset, and compare the following
models:

```python
import pandas as pd

df_hurricanes = pd.read_csv('data/frequency-north-atlantic-hurricanes.csv')

y_obs = df_hurricanes["Number of US Hurricanes (HUDRAT, NOAA)"].dropna().values

with pm.Model() as model_a:

    mu = pm.Gamma('mu', alpha=1, beta=1/10)

    p = pm.Poisson("y", mu, observed=y_obs)
    trace_a = pm.sample(draws=2000, tune=500, chains=4, return_inferencedata=True,
                        idata_kwargs = {'log_likelihood': True}, target_accept=0.9,
                       random_seed=np.random.default_rng(42))

with pm.Model() as model_b:
    mu = pm.Gamma('mu', alpha=1, beta=1/10)
    alpha = pm.Exponential('alpha', lam=0.1)
    p1 = pm.NegativeBinomial("y", mu=mu, n=alpha, observed=y_obs)
    trace_b = pm.sample(draws=2000, tune=500, chains=4, return_inferencedata=True,
                        idata_kwargs = {'log_likelihood': True}, target_accept=0.9,
                       random_seed=np.random.default_rng(42))
```

```python
az.plot_trace(trace_a)
```

![Trace hurricanes A](/docs/assets/images/model_comparison/trace_hurricanes_a.jpg)

```python
az.plot_trace(trace_b)
```

![Trace hurricanes B](/docs/assets/images/model_comparison/trace_hurricanes_b.jpg)

We can compute the LOO-WAIC as

```python
loo_a = az.loo(trace_a, model_a)
loo_b = az.loo(trace_b, model_b)

model_compare = az.compare({'Model a': loo_a, 'Model b': loo_b})
az.plot_compare(model_compare)
```

![LOO Plot](/docs/assets/images/model_comparison/loo.jpg)

Model $a$ is slightly preferred to model $a\,,$ as it is more accurate in reproducing
the data:

```python
ppc_a = pm.sample_posterior_predictive(trace_a, model_a)
ppc_b = pm.sample_posterior_predictive(trace_b, model_b)

fig = plt.figure()
ax = fig.add_subplot(211)
az.plot_ppc(ppc_a, ax=ax)
ax.set_xlim(0, 13)
ax1 = fig.add_subplot(212)
az.plot_ppc(ppc_b, ax=ax1)
ax1.set_xlim(0, 13)
```


![PPC Hurricanes](/docs/assets/images/model_comparison/ppc_hurricanes.jpg)

[^1]: See [here](https://statmodeling.stat.columbia.edu/2019/09/10/i-hate-bayes-factors-when-theyre-used-for-null-hypothesis-significance-testing/) or [here](https://vasishth.github.io/bayescogsci/book/ch-bf.html) and references therein.
[^2]: More precisely, they can be only consistently used with regular models, which are models where the posterior distribution can be asymptotically approximated with a normal distribution. See [Watanabe](https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf) for an in-depth discussion.
