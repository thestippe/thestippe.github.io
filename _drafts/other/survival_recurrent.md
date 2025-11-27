---
layout: post
title: "Survival analysis with recurrent events"
categories: /other/
up: /other
tags: /survival_recurrent/
# image: "/docs/assets/images/gis/point_patterns/posterior_xi.webp"
description: "Time to event analysis for repeated events"
date: "2025-01-28"
---

Up to now, when discussing survival analysis, we either assumed that the event
occurred exactly once or, when discussing 
[cure rate models](/statistics/survival_example_frailty),
that it could occur at most once.

We will now introduce time to event analysis in situations where the event
can show up more than once, and we will apply it to the colorectal
dataset from the [frailty R package](https://cran.r-project.org/web/packages/frailtypack/index.html),
and the corresponding rda file can be downloaded from
[here](https://github.com/socale/frailtypack/raw/refs/heads/master/data/colorectal.rda).

We will use a model from Karlin and Taylor, where the number of events
for each time interval $j=0,...,J-1$ for the individual $i$ is taken as:

$$
y_{ij} = \mathcal{Poisson}\left(w_i H_{0j} e^{\beta X_i}\right)
$$

where $w_i$ is the $i$-th individual's frailty and

$$
H_{0j} = \int_{t_j}^{t_{j+1}} dt' h(t')
$$

We will assume a B-spline model for $h(t)$, that we discussed in our
[dedicated post](/statistics/spline).


```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import seaborn as sns
from matplotlib import pyplot as plt

rng = sum(map(ord,'recurrent'))

kwargs = dict(
    draws=2000,
    tune=2000,
    chains=4,
    nuts_sampler='numpyro',
    random_seed=rng
)

df = pd.read_csv('colorectal.csv')

df.head()
```

|    |   Unnamed: 0 |   id |    time0 |    time1 |   new.lesions | treatment   | age         |   who.PS | prev.resection   |   state |   gap.time |
|---:|-------------:|-----:|---------:|---------:|--------------:|:------------|:------------|---------:|:-----------------|--------:|-----------:|
|  0 |            0 |    1 | 0        | 0.709589 |             0 | S           | 60-69 years |        0 | No               |       1 |  0.709589  |
|  1 |            1 |    2 | 0        | 1.28219  |             0 | C           | >69 years   |        0 | No               |       1 |  1.28219   |
|  2 |            2 |    3 | 0        | 0.52459  |             1 | S           | 60-69 years |        1 | No               |       0 |  0.52459   |
|  3 |            3 |    3 | 0.52459  | 0.920765 |             1 | S           | 60-69 years |        1 | No               |       0 |  0.396175  |
|  4 |            4 |    3 | 0.920765 | 0.942466 |             0 | S           | 60-69 years |        1 | No               |       1 |  0.0217007 |


As you can see, each id can appear more than once, and our aim is to quantify how
many events experiences a given id in a fixed amount of time.

Notice that the WHO Performance Status is a discrete score,
and it is in my opinion more appropriate to model it as a categorical
variable than as a discrete one, since we could equivalently label
the scores as A,B,C... without any loss of information.
We we don't do so, pandas will assume it is a numerical variable.

```python
df['who.PS'] = pd.Categorical(df['who.PS'])
```

In this context, a useful exploratory visualization is the abacus plot:

```python
fig, ax = plt.subplots(figsize=(7, 7))
for _id in df['id'].drop_duplicates():
    if not _id % 10:
        ax.axhline(y=_id, color='lightgray', lw=1.5, alpha=0.6)
    else:
        ax.axhline(y=_id, ls=':', color='lightgray', lw=1, alpha=0.6)
df.plot(x='time1', y='id', kind='scatter', marker='o', ax=ax)
ax.set_ylim(0, df['id'].max()+1)
ax.set_xlim(0, 4)
fig.tight_layout()
```

![The abacus plot for our dataset](/docs/assets/images/other/survival_recurrent/abacus.webp)

Before fitting the model, we must bring the dataset in wide format.
Since we are dealing with continuous time, we must first
construct the time bins.

```python
bins = np.arange(0, 3.9, 0.05)

df_gb = df.groupby(['id', pd.cut(df['time1'], bins=bins)]).count().unstack()['Unnamed: 0']

sns.heatmap(df_gb, cmap='cividis')
```

![The heatmap for the long format dataset](/docs/assets/images/other/survival_recurrent/heatmap.webp)



```python
p_fit = 2

knots = np.linspace(0, bins[-1], 12)
splines_dim_stop = len(knots)-p_fit-1
basis = np.array([bspline(bins[:-1], knots, i, p_fit) for i in range(splines_dim_stop)])

X = pd.get_dummies(df[['id', 'age', 'treatment', 'prev.resection', 'who.PS']
                   ].drop_duplicates().groupby('id').first(), drop_first=True
                   ).astype(int)

coords = {'id': df_gb.index, 'bins': bins, 'time': bins[:-1], 'weight': range(np.shape(basis)[0]), 'fac': X.columns}

with pm.Model(coords=coords) as model:
    X_ = pm.Data('X_', X, dims=('id', 'fac'))
    kappa = pm.Gamma('kappa', alpha=0.01, beta=0.01)
    t = pm.Data('t', basis, dims=('weight', 'time'))
    alpha = pm.Normal('alpha', mu=0, sigma=5, dims=('weight'))
    beta = pm.Normal('beta', mu=0, sigma=5, dims=('fac'))
    q = pm.Deterministic('q', pm.math.dot(z, t)*np.diff(bins)[0])
    h = pm.Deterministic('h', pm.math.exp(pm.math.cumsum(q))/np.diff(bins)[0], dims=('time'))
    H = pm.Deterministic('H', pm.math.cumsum(h)*np.diff(bins)[0], dims=('time'))
    rho = pm.Deterministic('rho', pm.math.dot(X_, beta), dims=('id'))
    w = pm.Gamma('w', alpha=1/kappa, beta=1/kappa, dims=('id'))
    y = pm.Poisson('y', mu=h[None, :]*np.diff(bins)[0]*w[:, None]*pm.math.exp(rho[:, None]), 
                   observed=df_gb, dims=('id', 'time'))

with model:
    idata = pm.sample(**kwargs)
    
az.plot_trace(idata, var_names=['w', 'alpha', 'beta'])
fig = plt.gcf()
fig.tight_layout()
```

![](/docs/assets/images/other/survival_recurrent/trace.webp)

The trace looks good, we can now inspect as usual the regression coefficients.

```python
fig, ax = plt.subplots()
az.plot_forest(idata, var_names=['beta'], combined=True, ax=ax)
fig.tight_layout()
```

![](/docs/assets/images/other/survival_recurrent/forest.webp)

```python
fig, ax = plt.subplots(figsize=(3, 13))
az.plot_forest(idata, var_names=['w'], combined=True, ax=ax)
ytks = ax.get_yticks()
ylab = ax.get_yticklabels()
ax.set_yticks(ytks[1::2])
ax.set_yticklabels(ylab[1::2])
fig.tight_layout()
```

![](/docs/assets/images/other/survival_recurrent/forest_w.webp)

As we could expect, there is a very large variability across the frailties.
We can now assess the temporal dependence of the intensity function

```python
az.plot_hdi(x=bins[:-1], y=idata.posterior['h'], smooth=False)
```

The intensity rapidly decreases as the time increases, as we can expect
by our initial plot.

![](/docs/assets/images/other/survival_recurrent/h.webp)

## Conclusions

In this post we have seen one of the many possible ways to model
generalize time to event data to recurrent events.

## Suggested readings

- <cite>Ibrahim, J. G., Chen, M., Sinha, D. (2013). Bayesian Survival Analysis. Springer New York.</cite>
- <cite>Karlin, S., Taylor, H. E. (1981). A Second Course in Stochastic Processes. US:Elsevier Science.</cite>

```python
%load_ext watermark

%watermark -n -u -v -iv -w -p xarray,numpyro,jax,jaxlib, pytensor
```

<div class="code">
<br>
Last updated: Tue Nov 18 2025<br>
<br>
Python implementation: CPython<br>
Python version       : 3.13.9<br>
IPython version      : 9.7.0<br>
<br>
xarray  : 2025.1.2<br>
numpyro : 0.19.0<br>
jax     : 0.8.0<br>
jaxlib  : 0.8.0<br>
pytensor: 2.35.1<br>
<br>
arviz     : 0.23.0.dev0<br>
numpy     : 2.3.4<br>
matplotlib: 3.10.7<br>
pandas    : 2.3.3<br>
pymc      : 5.26.1<br>
seaborn   : 0.13.2<br>
<br>
Watermark: 2.5.0
</div>