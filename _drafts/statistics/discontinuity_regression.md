---
layout: post
title: "Regression discontinuity "
categories: /statistics/
subgategory: "Causal inference"
tags: /discontinuity_regression/
date: "2024-02-11"
# image: "/docs/5ssets/images/perception/eye.jpg"
description: "Using jumps to estimate effects"
section: 4
---

Regression discontinuity recently became a popular way to assess the effect
of an intervention $I$ which depends on some confounder $X$ via

$$
I=
\begin{cases}
0 & x < x_0 \\
1 & x \geq x_0
\end{cases}
$$

where in general the effect of $Y$ on $X$ varies smoothly.
Since the dependence can be, in principle, arbitrary, many authors
discuss both the linear as well as higher order polynomials (See Angrist' textbook below).
However, higher order polynomial regression should in principle be avoided,
as it may lead to artificial discontinuities, as extensively explained
in [this work by Gelman and Imbens](http://www.stat.columbia.edu/~gelman/research/published/2018_gelman_jbes.pdf).

We will perform a bayesian version of [this analysis](https://lfoswald.github.io/2021-spring-stats2/materials/session-7/07-online-tutorial/) and, for the reasons explained
above, we will limit ourself to the linear dependence.
The dataset uses is the MADD dataset, which collects the
mortality rate of young people in the USA.

```python
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import seaborn as sns
from matplotlib import pyplot as plt

df_madd = pd.read_csv("https://raw.githubusercontent.com/seramirezruiz/stats-ii-lab/master/Session%206/data/mlda.csv")

fig = plt.figure()
ax = fig.add_subplot(111)
sns.scatterplot(data=df_madd,x='forcing', y='outcome')
ax.axvline(x=0, color='k', ls=':')
```

![The input data](/docs/assets/images/statistics/discontinuity_regression/data.webp)

Here "outcome" is the mortality rate from motor vehicle (per 100000),
while "forcing" is age minus 21 (we recall that, in the US, 21 is the age
where it is legally possible to drink alcoholic beverages).

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
  trace_madd = pm.sample(draws=2000, tune=500, chains=4)

az.plot_trace(trace_madd)
```

![The trace plot](/docs/assets/images/statistics/discontinuity_regression/trace.webp)

The trace looks good, let us verify if our model is able to reproduce the data:

```python
with madd_model:
    x_pl = np.arange(-2, 2, 1e-2)
    mu = alpha + beta*x_pl+gamma*np.heaviside(x_pl, 0.0)
    y_pl = pm.Normal('y_pl', mu=mu, sigma=sigma)
    pp_plot = pm.sample_posterior_predictive(trace=trace_madd, var_names=['y_pl'])

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

![The posterior predictive plot](/docs/assets/images/statistics/discontinuity_regression/posterior_predictive.webp)

The posterior predictive seems in good agreement with the observed data, and
the threshold effect seems quite evident. In fact it is quite large:

```python
az.plot_forest(trace_madd, filter_vars='like', var_names='gamma')
```

![The boxplot of the effect size](/docs/assets/images/statistics/discontinuity_regression/effect.webp)

## Conclusions

We have discussed how to implement the regression discontinuity,
together with some recommendations on how to implement it, and we applied
it to the MADD dataset. 

## Recommended readings
- <cite>Angrist, J. D., Pischke, J. (2009). Mostly harmless econometrics : an empiricist's companion. UK: Princeton University Press.
</cite>
