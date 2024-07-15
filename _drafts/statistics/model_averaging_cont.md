---
layout: post
title: "Model comparison, cont."
categories: /statistics/
subcategory: "Bayesian workflow"
tags: /model_comparison/
date: "2024-02-25"
section: 3
# image: "/docs/assets/images/perception/eye.jpg"
description: "Cross validation in Bayesian statistics"
---

We previously discussed the Bayes Factors as a tool to choose between different models.
This method has however many issues, and it is generally not recommended to use it.
We will now discuss a very powerful method, namely the Leave One Out cross validation


## Leave One Out cross-validation
This method is generally preferred to the above one, as it has been pointed out
that Bayes factors are appropriate only when one of the models is true,
while in real world problems we don't have any certainty about which is the model that
generated the data, assuming that it makes sense to claim that it exists such a model.
Moreover, the sampler used to compute the Bayes factor, namely Sequential Monte Carlo,
is generally less stable than the standard one used by PyMC, which is the NUTS sampler.
There are other, more philosophical reasons, pointed out by Gelman in [this post](
https://statmodeling.stat.columbia.edu/2017/07/21/bayes-factor-term-came-references-generally-hate/),
but for now we won't dig into this kind of discussion.

The LOO method is much more in the spirit of the Machine Learning, where
one splits the sample into a training set and a test set.
The train set is used to find the parameters, while the second one is
used to assess the performances of the model for new data.
This method, namely the **cross validation**, is by far the most
reliable one, and we generally recommend to use it.
It is however very common that the dataset is too small to allow
a full cross-validation.
The LOO cross validation is equivalent to the computation of

$$
ELPD = \sum_i \log p_{-i}(y_i)
$$

where $$p_{-i}(y_i)$$ is the posterior predictive probability
of the point $$y_i$$ relative to the model fitted by removing $$y_i\,.$$

We already anticipated this method in the post on the
[negative binomial model](/statistics/negbin),
but we will discuss it here more in depth.

In this example, we are looking for the distribution of the log-return
of an indian company.
We will first try and use a normal distribution. We will then use a more general
t-Student distribution to fit the data.

```python
import numpy as np
from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
import pandas as pd
from scipy.stats import beta
import yfinance as yf
import seaborn as sns

rng = np.random.default_rng(42)
tk = yf.Ticker("AJMERA.NS")

data = tk.get_shares_full(start="2023-01-01", end="2024-07-01")

logret = np.diff(np.log(data.values))

sns.histplot(logret, stat='density')
```
![](/docs/assets/images/statistics/model_averaging_cont/logret.webp)

The distribution shows heavy tails, it is therefore quite clear that a normal distribution
might not be appropriate.
We will however start from the simplest model, and use it as a benchmark for a more involved model

```python
with pm.Model() as norm:
    mu = pm.Normal('mu', mu=0, sigma=0.05)
    sigma = pm.Exponential('sigma', 1)
    yobs = pm.Normal('yobs', mu=mu, sigma=sigma, observed=logret)

with norm:
    idata_norm = pm.sample(nuts_sampler='numpyro', random_seed=rng)

az.plot_trace(idata_norm)
fig = plt.gcf()
fig.tight_layout()
```
![](/docs/assets/images/statistics/model_averaging_cont/trace_norm.webp)


The trace doesn't show any issue. Let us try with a Student-T distribution

```python
with pm.Model() as t:
    mu = pm.Normal('mu', mu=0, sigma=0.05)
    sigma = pm.Exponential('sigma', 1)
    nu = pm.Gamma('nu', 10, 0.1)
    yobs = pm.StudentT('yobs', mu=mu, sigma=sigma, nu=nu, observed=logret)

with t:
    idata_t = pm.sample(nuts_sampler='numpyro', random_seed=rng)

az.plot_trace(idata_t)
fig = plt.gcf()
fig.tight_layout()

```
![](/docs/assets/images/statistics/model_averaging_cont/trace_t.webp)

Also in this case the trace doesn't show any relevant issue.
Let us now check if we are able to reproduce the observed data.

```python
with norm:
    idata_norm.extend(pm.sample_posterior_predictive(idata_norm, random_seed=rng))

fig = plt.figure()
ax = fig.add_subplot(111)
az.plot_ppc(idata_norm, num_pp_samples=500, ax=ax, mean=False)
```

![](/docs/assets/images/statistics/model_averaging_cont/ppc_norm.webp)

Our model seems totally unable to fit the data due to the presence of heavy tails.
Let us now verify if the second model does a better job.

```python
with t:
    idata_t.extend(pm.sample_posterior_predictive(idata_t, random_seed=rng))

fig = plt.figure()
ax = fig.add_subplot(111)
az.plot_ppc(idata_t, num_pp_samples=500, ax=ax, mean=False)
ax.set_xlim([-0.1, 0.1])
```

![](/docs/assets/images/statistics/model_averaging_cont/ppc_t.webp)

As you can see, there is much more agreement with the data, so this model looks
more appropriate.
Let us now see if the LOO cross validation confirms our first impression.

```python
with norm:
    pm.compute_log_likelihood(idata_norm)

with t:
    pm.compute_log_likelihood(idata_t)

df_comp_loo = az.compare({"norm": idata_norm, "t": idata_t})

az.plot_compare(df_comp_loo)
```

![](/docs/assets/images/statistics/model_averaging_cont/loo.webp)

```python
df_comp_loo
```

|      |   rank |   elpd_loo |   p_loo |   elpd_diff |   weight |      se |     dse | warning   | scale   |
|:-----|-------:|-----------:|--------:|------------:|---------:|--------:|--------:|:----------|:--------|
| t    |      0 |    929.203 | 2.6402  |       0     | 0.870715 | 30.6267 |  0      | False     | log     |
| norm |      1 |    740.195 | 7.50734 |     189.008 | 0.129285 | 30.2578 | 25.8493 | False     | log     |


```python
az.plot_loo_pit(idata_norm, y="yobs", ecdf=True)
```
![](/docs/assets/images/statistics/model_averaging_cont/loo_pit_norm.webp)


```python
az.plot_loo_pit(idata_t, y="yobs", ecdf=True)
```
![](/docs/assets/images/statistics/model_averaging_cont/loo_pit_t.webp)

```python
%load_ext watermark
```

```python
%watermark -n -u -v -iv -w -p xarray,pytensor,numpyro,jax,jaxlib
```
<div class="code">
Last updated: Mon Jul 08 2024
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
numpyro : 0.15.0
<br>
jax     : 0.4.28
<br>
jaxlib  : 0.4.28
<br>
<br>
pymc      : 5.15.0
<br>
matplotlib: 3.9.0
<br>
arviz     : 0.18.0
<br>
numpy     : 1.26.4
<br>
pandas    : 2.2.2
<br>
yfinance  : 0.2.40
<br>
seaborn   : 0.13.2
<br>
<br>
Watermark: 2.4.3
</div>