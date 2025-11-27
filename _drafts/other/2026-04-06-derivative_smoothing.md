---
categories: /other/
up: /other
date: 2026-04-06
description: Applying bayesian statistics to functional data analysis
layout: post
section: 10
subcategory: Other random thoughts
tags: /complex/
title: Derivative smoothing
---

There are circumstances when it is more natural to think about your data
as describing a function encoding some specific property, and functional
data analysis is the statistics branch dealing with this kind of data.
It shares many common concepts with nonparametric statistics, but
many authors consider it as a separate field.
One of the core concepts is functional data analysis is how to reliably find
the derivative of a function, and here we will show how to use PyMC
to do so.

We will use the Berkeley growth dataset, which describes the
length of a group of 93 children from the age of 6 months up to
18 years.
The dataset can be downloaded by using R from the fda package by running,
in a Jupyter-Lab session running with an R kernel,

```R
install.packages("fda")
library('fda')
data('growth')
```

We will try and find out the growth velocity for the males of the study,
so rather than focusing on $y(t)$, we will try and find
a way to represent its derivative $y'(t).$

A way to do so could be to simply compute

$$
y(t) = y(t_0) + \int_{t_0}^t dt' y'(t')
$$

but we can do better, by requiring that the measured length is a positive
quantity.
The usual way to enforce positiveness is by parametrizing

$$
y(t) = e^{W(t)}.
$$

We can combine the two representations by defining

$$
\frac{y'(t)}{y(t)} = \frac{d}{dt} log\left(y(y)\right) = w(t)
$$

so

$$
\log\left(y(y)\right) = C + \int_{t_0}^t dt' w(t')
$$

or, in other terms,

$$
y(t) = y(t_0) \exp\left(\int_{t_0}^t dt' w(t')\right)\,.
$$

This representation allows us to immediately compute
$ y'(t) = w(t) y(t) \,.$

In our case it makes sense to assume that the first derivative is non-negative too,
since it's quite natural to assume that the body length does not decrease up to 20 years
or so, and we will therefore assume
$$
w(t) = e^{\xi(t)}.
$$

We need some kind of parametrization for the growth function of the i-th
individual $\xi_i(t)$, and we will use a hierarchical spline model.
In this way, we will obtain the average growth function for free, and
we will be able to quantify the variability of the growth function
across individuals.

```python
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt

rng = np.random.default_rng(20251026)

df_fit = pd.read_csv('growth_m.csv')

df_fit['age'] /= 10
df_fit[df_fit.columns[1:]] /= 100

fig, ax = plt.subplots()

for col in df_m.columns[1:]:
    ax.plot(df_fit['age'], df_fit[col], color='lightgray')
```

![The Berkeley growth dataset](/docs/assets/images/statistics/derivative_smoothing/data.webp)

```python
yobs = df_fit[df_fit.columns[1:]]

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

We will use a spline with 16 knots.

```python
nx = 16
x_fit = np.array([0]+list(np.quantile(df_fit['age'].values, np.linspace(0, 1., nx)[1:-1]))+[2])

p_fit = 2

splines_dim = len(x_fit)-p_fit-1

basis = np.array([bspline(df_fit['age'].values, x_fit, i, p_fit) for i in range(splines_dim)])

coords = {'ind': yobs.columns, 'age': yobs.index, 'w_dim': range(splines_dim)}

with pm.Model(coords=coords) as model:
    sigma = pm.HalfNormal('sigma', 1)
    dx = pm.Data('dx', df_fit['age'].diff().fillna(0).values, dims=('age'))
    z = pm.Data('z', basis, dims=('w_dim', 'age'))
    alpha_mu = pm.Normal('alpha_mu', mu=0, sigma=0.5)
    alpha_sigma = pm.Exponential('alpha_sigma', 1)
    alpha_std = pm.Normal('alpha_std', mu=0, sigma=1, dims=('ind'))
    alpha = pm.Deterministic('alpha', alpha_mu + alpha_sigma*alpha_std, dims=('ind'))
    beta = pm.Normal('beta', mu=0, sigma=10, dims=('w_dim'))
    gamma = pm.Exponential('gamma', lam=0.5, dims=('w_dim'))
    w_std = pm.Normal('w_std', mu=0, sigma=1, dims=('ind', 'w_dim'))
    w = pm.Deterministic('w', beta+gamma*w_std, dims=('ind', 'w_dim'))
    xi = pm.Deterministic('xi', pm.math.exp(pm.math.dot(w, z)), dims=('ind', 'age'))  # the individual growth function
    mu = pm.Deterministic('mu', pm.math.exp(alpha[None, :] + pm.math.cumsum(xi.T*dx[:, None], axis=0)), dims=('age', 'ind'))
    y = pm.Normal('y', mu=mu, sigma=sigma, dims=('age', 'ind'), observed=yobs)

pm.model_to_graphviz(model)
```

![](/docs/assets/images/statistics/derivative_smoothing/model.webp)

The model is looks quite involved, but we can however identify two components:
$\alpha_i$, which is the logarithm of the length at the time of the first measurement,
and $\xi.$

Notice that the first component of $dx$ is 0, since it's equal to the first component
of 
`df_fit['age'].diff().fillna(0)`
, so the first component of
$\xi$ will be unconstrained. We will however keep this parametrization
for its extreme simplicity.

```python
with model:
    idata = pm.sample(**kwargs)

az.plot_trace(idata, var_names=['alpha_mu', 'alpha_sigma', 'sigma',  'beta', 'gamma'])
fig = plt.gcf()
fig.tight_layout()
```

![](/docs/assets/images/statistics/derivative_smoothing/trace.webp)


Everything looks fine, so we can take a look at the result of our fit.

```python
with model:
    idata.extend(pm.sample_posterior_predictive(idata))


nr = 5
nc = 8
fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(12, 9), sharex=True, sharey=True)
sel_list = df_m.columns[1:]
for i in range(nr):
    for j in range(nc):
        k = i*nc+j
        if k < len(sel_list):
            ax[i][j].scatter(df_m['age'], df_m[sel_list[k]]/100, color='lightgray', alpha=0.7, marker='x')
            az.plot_hdi(df_m['age'].drop_duplicates(), idata.posterior_predictive['y'].sel(ind=sel_list[k]), ax=ax[i][j])
            ax[i][j].set_title(sel_list[k])
        
fig.tight_layout()
```

![The comparison of the posterior predictive with the data](
/docs/assets/images/statistics/derivative_smoothing/fit.webp)

It looks like our fit accurately interpolates the observed data.
Let us now compare the derivative as naively computed as the ratio
between the difference of the length divided by the difference of the age
with our smoothed version.

```python
fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(12, 9), sharex=True, sharey=True)

for i in range(nr):
    for j in range(nc):
        k = i*nc+j
        if k < len(sel_list):
            az.plot_hdi(df_m['age'].drop_duplicates(), idata.posterior['xi'].sel(ind=sel_list[k])*idata.posterior['mu'].sel(ind=sel_list[k]), ax=ax[i][j])
            ax[i][j].scatter(df_m['age'][1:], df_m[sel_list[k]].diff()[1:]/df_m['age'].diff()[1:]/10, marker='x', color='lightgray')
            #ax[i][j].
            ax[i][j].set_title(sel_list[k])
            ax[0][0].set_ylim([0, 2])
            ax[0][0].set_xlim([1, 18])
fig.tight_layout()
```

![](/docs/assets/images/statistics/derivative_smoothing/derivative.webp)

Our fit is compatible with the data, except for some deviations
due to the smoothness requirement.

## Conclusions

We have seen how to find a smooth interpolation of the derivative of a function
by using a differential representation of the function, and how
to impose constraints on the function itself as well as on its derivatives.
We finally applied the above methods to a subset of the  Berkeley growth dataset.

## Suggested readings

- <cite>Ramsay, J., Silverman, B. W. (2005). Functional Data Analysis. Germania: Springer New York.</cite>


```python
%load_ext watermark
```

```python
%watermark -n -u -v -iv -w -p xarray,numpyro
```

<div class="code">
Last updated: Sun Oct 26 2025<br>
<br>
Python implementation: CPython<br>
Python version       : 3.12.8<br>
IPython version      : 8.31.0<br>
<br>
xarray : 2025.10.1<br>
numpyro: 0.16.1<br>
<br>
pymc      : 5.26.1<br>
numpy     : 2.3.4<br>
arviz     : 0.22.0<br>
pandas    : 2.3.3<br>
matplotlib: 3.10.7<br>
<br>
Watermark: 2.5.0
</div>