---
layout: post
title: "The Gaussian model"
categories: /statistics/
tags: /reals/
subcategory: "Simple models"
date: "2024-01-16"
section: 3
# image: "/docs/assets/images/perception/eye.jpg"
description: "Handling real-valued data"
---

In the previous post we worked with discrete
data (either binary or count data).

Before moving to more advanced topic, we will briefly show that
dealing with real-valued data only requires
to use a real-valued likelihood.

## The normal distribution

In order to show how to deal with real-valued data, we will
use a well known dataset which is well described by a normal distribution,
namely the housefly wing dataset provided at [this page](https://seattlecentral.edu/qelp/sets/057/057.html).

As reported in the linked page, the dataset contains the measurements of the wing length
for a set of housefly expressed in units of $0.1$ mm.

When performing Bayesian statistics, as well as almost any other numerical
computation, it is crucial to properly scale the data in order to avoid numerical
issues such as overflow or underflow.
An appropriate scaling has another advantage, which is making easier to estimate the order of magnitude of the priors.
We will assume a normal distribution likelihood with mean $\mu$ and variance $\sigma^2$
for the data expressed in cm.
We do expect that the housefly has a wing length of few millimeters,
so assuming

$$
\mu \sim \mathcal{N}(0, 1)
$$

and

$$
\sigma \sim \mathcal{Exp}(1)
$$

should be a generous enough guess for our priors.

```python
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
from matplotlib import pyplot as plt

df_length = pd.read_csv('https://seattlecentral.edu/qelp/sets/057/s057.txt',
                        header=None).rename(columns={0: 'wing_length'})

rng = np.random.default_rng(42)

with pm.Model() as model:
    sig = pm.Exponential('sig', lam=1)
    mu = pm.Normal('mu', mu=0, sigma=1)
    y = pm.Normal('y', mu=mu, sigma=sig, observed=df_length['wing_length']/100)
    trace = pm.sample(random_seed=rng)

az.plot_trace(trace)
fig = plt.gcf()
fig.tight_layout()
```

![The trace for the normal model](/docs/assets/images/statistics/reals/trace_norm.webp)

The trace looks fine, as usual.

Since we are dealing with two variables, looking at the single marginal distributions
only gives us some partial information about the structure of the posterior.
We can gain some information by looking at the joint distribution as follows

```python
az.plot_pair(trace, var_names=['sig', 'mu'],
            kind='kde')
```

![The joint posterior density](/docs/assets/images/statistics/reals/kde.webp)

Let us verify our posterior predictive.

```python
with model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=rng)

az.plot_ppc(ppc)
```

![The posterior predictive for the normal model](/docs/assets/images/statistics/reals/ppc_norm.webp)

As expected, the agreement is almost perfect.

## Conclusions

We showed that dealing with real-valued data requires no modifications to the method discussed
in the previous posts.
Starting from the next post, we will discuss how to deal with more advanced models.

```python
%load_ext watermark
```

```python
%watermark -n -u -v -iv -w -p xarray,pytensor
```
<div class="code">
Last updated: Tue Jun 25 2024
<br>

<br>
Python implementation: CPython
<br>
Python version       : 3.12.4
<br>
IPython version      : 8.24.0
<br>

<br>
xarray  : 2024.5.0
<br>
pytensor: 2.20.0
<br>

<br>
matplotlib: 3.9.0
<br>
arviz     : 0.18.0
<br>
pandas    : 2.2.2
<br>
pymc      : 5.15.0
<br>
numpy     : 1.26.4
<br>

<br>
Watermark: 2.4.3
<br>
</div>