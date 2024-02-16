---
layout: post
title: "Poisson regression"
categories: /statistics/
subgategory: "Regression"
tags: /logistic_regression/
section: 3
# image: "/docs/assets/images/perception/eye.jpg"
description: "Regression on count data"
date: "2024-01-28"
---

In the last post we introduced the Generalized Linear Models,
and we explained how to perform regression on data types which are
not appropriate for a Gaussian likelihood.
We also saw a concrete example of logistic regression, and here we will
discuss another type of GLM, the Poisson regression.

## Poisson regression

In the Poisson regression one assumes that

$$
Y_i \sim \mathcal{Poisson}(\theta_i)
$$

where $\theta_i$ must be a non-negative variable. One can 
use the exponential function to map any real number on the
positive axis, we therefore assume that

$$
\theta_i = \exp\left(\alpha + \beta X_i\right)
$$

We will use this model to estimate the average number of
bear attacks in North America.
The original data can be found on this [data.world
](https://data.world/ajsanne/north-america-bear-killings/workspace/file?filename=north_america_bear_killings.csv)
page, where there are listed all human killing by a black, brown, or polar bear from 1900-2018 in North America.
We will limit ourself to black and brown bears, as attacks by polar bears are very rare.
We will also limit our dataset to the years after 1999, as we want to assume that the attack probability
is constant within the entire time range, and we will neglect attacks by captive animals.
We want to assess the attack probability by bear type, and to do this we will use the bear type
as a regressor.

```python
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('./data/north_america_bear_killings.csv')

rng = np.random.default_rng(seed=42)

df_red = df.groupby(['Year', 'Type', 'Type of bear']).count().reset_index()[['Year', 'Type', 'Type of bear', 'Hikers']]

df_clean = df_red[(df_red['Year']>=2000) & (df_red['Type of bear'] != 'Polar Bear')& (df_red['Type'] != 'Captive')]

df_fit = df_clean.set_index(['Year', 'Type of bear']).unstack(fill_value=0).stack().reset_index()[['Year', 'Type of bear', 'Hikers']]

df_fit.rename(columns={'Hikers': 'Count'}, inplace=True)

df_fit['is_black'] = df_fit['Type of bear'].str.contains('Black').astype(int)

with pm.Model() as poisson_regr:
    alpha = pm.Normal('alpha', mu=0, sigma=2)
    beta = pm.Normal('beta', mu=0, sigma=2)
    z = pm.math.exp(alpha + beta*df_fit['is_black'])
    y = pm.Poisson('y', mu=z, observed=df_fit['Count'])
    y_brown = pm.Poisson('y_brown', mu=pm.math.exp(alpha))
    y_black = pm.Poisson('y_black', mu=pm.math.exp(alpha+beta))

with poisson_regr:
    trace = pm.sample(draws=2000, tune=2000, random_seed=rng)

az.plot_trace(trace)
```

![The trace of the Poisson model](/docs/assets/images/statistics/poisson_glm/trace.webp)

The trace seems fine, we can now verify if the model is compatible with the data.

```python
with poisson_regr:
    ppc = pm.sample_posterior_predictive(trace, var_names=['y', 'y_brown', 'y_black'], random_seed=rng)

# Let us estimate the probability that, in one year, there are k brown/black bear attacks 

n_brown = np.array([np.count_nonzero(ppc.posterior_predictive['y_brown'].values.reshape(-1)==k) for k in range(10)])
n_black = np.array([np.count_nonzero(ppc.posterior_predictive['y_black'].values.reshape(-1)==k) for k in range(10)])

fig = plt.figure()
ax = fig.add_subplot(211)
ax.hist(df_fit[df_fit['is_black']==0]['Count'], alpha=0.5, color='crimson', density=True, bins=np.arange(10), width=0.8)
ax.bar(np.arange(10)+0.3, n_brown/n_brown.sum(), alpha=0.7, width=0.8)
ax1 = fig.add_subplot(212)
ax1.hist(df_fit[df_fit['is_black']==1]['Count'], alpha=0.5, color='crimson', density=True, bins=np.arange(10), width=0.8)
ax1.bar(np.arange(10)+0.3, n_black/n_black.sum(), alpha=0.7, width=0.8)
```

![The posterior predictive of the Poisson model](/docs/assets/images/statistics/poisson_glm/posterior_predictive.webp)

The data seems compatible with the average estimate of our model.
We can now verify if the average number of attacks by black bears is statistically
compatible with the average number of attacks by brown bears.

```python
az.plot_forest(trace, var_names=['alpha', 'beta'])
```

![The forest plot of our parameters](/docs/assets/images/statistics/poisson_glm/posterior.webp)

As we can see, $\beta$ is compatible with 0, so we can consider the average attack number
by black bears is compatible with the average attack number by brown bears.

## Conclusions

We discussed a second kind of GLM, namely the Poisson regression,
and we applied this model to estimate the average number of lethal
attacks by wild bears in North America.
