---
layout: post
title: "Regression discontinuity design"
categories: /statistics/
subcategory: "Causal inference"
tags: /discontinuity_regression/
date: "2024-07-07"
# image: "/docs/5ssets/images/perception/eye.jpg"
description: "Introducing an arbitrary threshold to infer causality"
section: 5
---

Regression Discontinuity Design (RDD) can be applied when there is a threshold
above which some causal effect applies, and allows you to infer the impact of such an effect
on your population.
In most countries, there is a retirement age, and you might analyze the impact of the
retirement on your lifestyle.
There are also countries where school classes has a maximum number of students,
and this has been used to assess the impact of the number of students on the students' performances.
Here we will re-analyze, in a Bayesian way, the impact of alcohol on the mortality, as done in "Mastering Metrics".
In the US, at 21, you are legally allowed to drink alcohol,
and we will use RDD to assess the impact on this on the probability of death in the US.

## Implementation

Let us first of all take a look at the dataset.

```python
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import seaborn as sns
from matplotlib import pyplot as plt

df_madd = pd.read_csv("https://raw.githubusercontent.com/seramirezruiz/stats-ii-lab/master/Session%206/data/mlda.csv")

rng = np.random.default_rng(42)

fig = plt.figure()
ax = fig.add_subplot(111)
sns.scatterplot(data=df_madd,x='forcing', y='outcome')
ax.axvline(x=0, color='k', ls=':')
fig.tight_layout()
```

![](/docs/assets/images/statistics/rdd/data.webp)

A linear model seems appropriate, and it seems quite clear that there is a jump when
the forcing variable (age-21) is zero.

While RDD can be both applied with a sharp cutoff and a fuzzy one, we will
limit our discussion to the sharp one.
We will take a simple linear model, as [polynomial models should be generally avoided in RDD models](https://stat.columbia.edu/~gelman/research/published/2018_gelman_jbes.pdf)
as they tend to introduce artifacts.

$$
y \sim \mathcal{N}( \alpha + \beta x + \gamma \theta(x), \sigma)
$$

Here $x$ is the age minus 21, while $\theta(x)$ is the Heaviside theta

$$
\theta(x)
=
\begin{cases}
0 & x\leq0 \\
1 & x > 0\\
\end{cases}
$$

As usual, we will assume a non-informative prior for all the parameters.

```python
with pm.Model() as madd_model:
  alpha = pm.Normal('alpha', mu=0, sigma=1000)
  gamma = pm.Normal('gamma', mu=0, sigma=1000)
  beta = pm.Normal('beta', mu=0, sigma=100)
  mu_0 = alpha + beta*df_madd['forcing'].values
  mu = mu_0 + gamma*np.heaviside(df_madd['forcing'].values, 0.0)
  sigma = pm.HalfCauchy('sigma', beta=5)
  y = pm.Normal('y', mu=mu, sigma=sigma, 
                observed=df_madd['outcome'].values)
  idata = pm.sample(draws=5000, tune=5000, chains=4, nuts_sampler='numpyro', random_seed=rng)

az.plot_trace(idata)
fig_trace = plt.gcf()
fig_trace.tight_layout()
```

![](/docs/assets/images/statistics/rdd/trace.webp)

The trace looks fine, and it is clear that the value of the discontinuity is quite large.

```python
az.plot_forest(idata, filter_vars='like', var_names='gamma')
```
![](/docs/assets/images/statistics/rdd/effect.webp)

Let us now verify if our model is capable of reproducing the observed data.

```python
with madd_model:
    x_pl = np.arange(-2, 2, 1e-2)
    mu = alpha + beta*x_pl+gamma*np.heaviside(x_pl, 0.0)
    y_pl = pm.Normal('y_pl', mu=mu, sigma=sigma)
    pp_plot = pm.sample_posterior_predictive(trace=idata, var_names=['y_pl'], random_seed=rng)

pp_madd = pp_plot.posterior_predictive.y_pl.values.reshape((-1, len(x_pl)))

madd_mean = np.mean(pp_madd, axis=0)
madd_qqmax = np.quantile(pp_madd,0.975, axis=0)
madd_qqmin = np.quantile(pp_madd,0.025, axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
sns.scatterplot(data=df_madd,x='forcing', y='outcome')
ax.axvline(x=0, color='k', ls=':')
ax.plot(x_pl, madd_mean, color='k')
ax.fill_between(x_pl, madd_qqmin, madd_qqmax,
                color='grey', alpha=0.5)
ax.set_xlim([-2, 2])
```

![](/docs/assets/images/statistics/rdd/posterior_predictive.webp)

## Conclusions
We re-analyzed the effect of the Minimum Legal Driving Age (MLDA)
on the mortality, and we discussed how to apply RDD to perform causal inference
in the presence of a threshold.


## Suggested readings

- <cite>Imbens, G. W., Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction. US: Cambridge University Press.<cite>
- <cite><a href='https://arxiv.org/pdf/2206.15460.pdf'>Li, Ding, Mealli (2022). Bayesian Causal Inference: A Critical Review</a></cite>
- <cite>Ding, P. (2024). A First Course in Causal Inference. CRC Press.</cite>
- <cite>Angrist, J. D., Pischke, J. (2014). Mastering 'Metrics: The Path from Cause to Effect.   Princeton University Press.</cite>

```python
%load_ext watermark
```


```python
%watermark -n -u -v -iv -w -p xarray,pytensor
```

<div class="code">
Last updated: Tue Aug 20 2024
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
seaborn   : 0.13.2
<br>
pandas    : 2.2.2
<br>
arviz     : 0.18.0
<br>
pymc      : 5.15.0
<br>
numpy     : 1.26.4
<br>

<br>
Watermark: 2.4.3
<br>
</div>