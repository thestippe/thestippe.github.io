---
layout: post
title: "Splines regression"
categories: /statistics/
subcategory: "Non-parametric models"
tags: /splines_regression/
date: "2024-06-03"
section: 7
# image: "/docs/assets/images/perception/eye.jpg"
description: "Going beyond linear models"
---

A common issue that any data scientist faced at some point
is how to include non-linearities into regression.
It is well known that higher order polynomials tend to overfit,
and it is therefore generally not a good idea to use this kind of
model.

A very flexible solution is given by the splines, which are
piecewise smooth functions.
There are many kind of splines, and recently 
[B-splines](https://en.wikipedia.org/wiki/B-spline) gained
a lot of attention since they are very easy to implement, and they
are numerically stable.
Here we will use them to fit the "Six cities study"
of [this link](https://content.sph.harvard.edu/fitzmaur/ala2e/),
from the "Applied Longitudinal Analysis" textbook by Fitzmaurice, Laird
and Ware.

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

df = pd.read_stata('fev1.dta')

df['id'] = df['id'].astype(int)

sns.scatterplot(df, x='age', y='logfev1')
```
![Our dataset](/docs/assets/images/statistics/splines/data.webp)

We will try and model the relation between the Forced Expiratory Volume
(FEV) and the age.
The relation seems linear up to roughly 16 years, while above this point
it looks like the FEV saturates.
This is quite clear, since at some point the breath capacity must
saturate, since our growth stops.
Let us now implement the B-splines according to Wikipedia's recursive algorithm

```python
def bspline(t: np.array, x:np.array, i:int, p:int) -> float:
    """
    Returns a B-spline, defined as https://en.wikipedia.org/wiki/B-spline.

    Parameters:
    -----------
    t: np.array
    x: np.array
    i: int
    p: int

    Returns:
    np.array
    
    Raises:
    ------
    ValueError
       if i is not an integer between 0 and len(x)-p-1 (both included)
    """
    if i>=0 and i<len(x)-p-1:
        if p==0:
            return np.heaviside(t-x[i], 1)*np.heaviside(x[i+1]-t, 0)
        else:
            fac0 = (t-x[i])/(x[i+p]-x[i])
            fac1 = (x[i+p+1]-t)/(x[i+p+1]-x[i+1])
            return (fac0*bspline(t, x, i, p-1)+fac1*bspline(t, x, i+1, p-1))
    else:
        raise ValueError(f'Got i={i}, i must be an integer between 0 and len(x)-p-1={len(x)-p-1}')
```

The splines are quite fast, but our dataset contains thousands of points,
and for each point we calculate the likelihood thousands of times,
so it is better to precompute the value of the splines.
We will use second order splines in order to ensure smoothness,
and we will use 15 knots.


```python
x_fit = np.linspace(5, 20, 15)
p_fit = 2
splines_dim = len(x_fit)-p_fit-1

basis = np.array([bspline(df['age'].values, x_fit, i, p_fit) for i in range(splines_dim)])
```

Before fitting the model, let us take a look at our basis functions

```python
x_plot = np.arange(5, 20, 15)

basis_plot = np.array([bspline(x_plot, x_fit, i, p_fit) for i in range(splines_dim)])

fig = plt.figure()
ax = fig.add_subplot(111)
for elem in basis_plot:
    ax.plot(x_plot, elem)
```

![Our basis functions](/docs/assets/images/statistics/splines/basis.webp)

We are finally ready to fit our model

```python
with pm.Model() as spline_model:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1)
    w = pm.Normal('w', mu=0, sigma=20, shape=(splines_dim))
    sigma = pm.Exponential('sigma', lam=1)
    mu = alpha + beta*df['age'] + pm.math.dot(w, basis)
    yhat = pm.Normal('yhat', mu=mu, sigma=sigma, observed=df['logfev1'])

with spline_model:
    idata_spline = pm.sample(nuts_sampler='numpyro', draws=5000, target_accept=0.9)

az.plot_trace(idata_spline)
fig = plt.gcf()
fig.tight_layout()
```

![Our dataset](/docs/assets/images/statistics/splines/trace.webp)

We can now verify if our model is able to describe the data.


```python
with spline_model:
    mu_pred = pm.Deterministic('mu_pred', alpha + beta*x_plot + pm.math.dot(w, basis_plot))
    yhat_pred = pm.Normal('yhat_pred', mu=mu_pred, sigma=sigma)

with spline_model:
    ppc_spline = pm.sample_posterior_predictive(idata_spline, var_names=['yhat_pred', 'mu_pred'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_plot,
ppc_spline.posterior_predictive['yhat_pred'].mean(dim=['draw', 'chain']))
ax.fill_between(x_plot,
ppc_spline.posterior_predictive['yhat_pred'].quantile(q=0.025, dim=['draw', 'chain']),
ppc_spline.posterior_predictive['yhat_pred'].quantile(q=0.975, dim=['draw', 'chain']),
                color='lightgray', alpha=0.8
               )
ax.scatter(df['age'], df['logfev1'], color='gray', alpha=0.8)
ax.set_xlim([x_plot[0], x_plot[-1]])
fig.tight_layout()
```

![Our dataset](/docs/assets/images/statistics/splines/ppc.webp)

As we can see, our model accurately reproduces our data without
overfitting it. It has some small issue above 18, and the fact
that splines are not very stable just below the boundaries
is a known issue.

## Conclusions

You should consider using splines if you need more flexibility than
ordinary linear regression, as they allow you to smoothly add non-linearity
without overfitting.