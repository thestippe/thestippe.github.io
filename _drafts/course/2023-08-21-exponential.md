---
layout: post
title: "The exponential model"
categories: course/intro/
tags: /exponential/
image: "/docs/assets/images/exponential/events.png"
description: "Describing waiting times and other positive quantities"
---

In the [last](/normal/) post we used the normal and the Student-t distribution
to fit data belonging to the entire real axis.
You may however encounter data which are real but positive,
like a waiting time, and in this case it is wise not to allow to your predicted
data to be negative.

The **exponential** distribution is the simplest distribution
with this constraint, and it has as pdf

$$
p(y \vert \lambda) = \lambda e^{- \lambda y} \theta(y)\,.
$$

The parameter $\lambda$ represents the inverse of the mean of the distribution,
which is equal to its variance, while $\theta(y)$ is the Heaviside theta distribution,
which is equal to 1 if $y>0$ and its value is 0 otherwise.

In order to show an application to this model, I downloaded from
the [Italian Geology and Volcanology Institute](http://cnt.rm.ingv.it/)
the dataset containing all the seismic events of the last 90 days

```python
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt
import pymc.sampling_jax as pmjax

plt.style.use("seaborn-v0_8-darkgrid")

rng = np.random.default_rng(42)

df = pd.read_csv('data/earthqakes.csv', sep='|')

df.head()
```

|    |   #EventID | Time                       |   Latitude |   Longitude |   Depth/Km | Author                           |   Catalog |   Contributor |   ContributorID | MagType   |   Magnitude | MagAuthor   | EventLocationName                      | EventType   |
|---:|-----------:|:---------------------------|-----------:|------------:|-----------:|:---------------------------------|----------:|--------------:|----------------:|:----------|------------:|:------------|:---------------------------------------|:------------|
|  0 |   35633631 | 2023-07-24 00:30:05.420000 |    42.8808 |     13.0897 |       10.8 | SURVEY-INGV                      |       nan |           nan |             nan | ML        |         0.9 | --          | 4 km E Preci (PG)                      | earthquake  |
|  1 |   35633671 | 2023-07-24 00:51:50.980000 |    39.859  |     15.3635 |       10   | SURVEY-INGV                      |       nan |           nan |             nan | ML        |         2.1 | --          | Golfo di Policastro (Salerno, Potenza) | earthquake  |
|  2 |   35634111 | 2023-07-24 02:55:44.630000 |    37.743  |     15.068  |        3.8 | SURVEY-INGV-CT#SeismPicker_SO-OE |       nan |           nan |             nan | ML        |         2.5 | --          | 5 km W Milo (CT)                       | earthquake  |
|  3 |   35633921 | 2023-07-24 03:01:19.100000 |    37.742  |     15.061  |        4   | SURVEY-INGV-CT                   |       nan |           nan |             nan | ML        |         3.1 | --          | 5 km W Milo (CT)                       | earthquake  |
|  4 |   35634091 | 2023-07-24 03:08:10.740000 |    43.437  |     12.6908 |       12.6 | SURVEY-INGV                      |       nan |           nan |             nan | ML        |         0.6 | --          | 4 km NE Scheggia e Pascelupo (PG)      | earthquake  |

```python
df['Time'] = pd.to_datetime(df['Time'])
```

Let us give a look to the distribution of the waiting time between two subsequent
events

```python
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.hist(df['Time'].diff().dropna().dt.total_seconds()/360, bins=np.arange(0, 48, 2), density=True)
```

![Waiting time between two events](/docs/assets/images/exponential/events.png)

The distribution looks like an exponential, as expected.
Let us now try and fit the data

```python
delta_hours = df['Time'].diff().dropna().dt.total_seconds().values/360

with pm.Model() as eq_model:
    lam = pm.Gamma('lam', alpha=0.1, beta=0.1)
    y = pm.Exponential('y', lam=lam, observed=delta_hours)
    trace_eq = pmjax.sample_numpyro_nuts(draws=2000, tune=2000, chains=4, random_seed=rng)

az.plot_trace(trace_eq)
```

![Our trace](/docs/assets/images/exponential/trace.png)

The trace looks OK, let us check if we are able to reproduce the data:

```python
with eq_model:
    ppc_eq = pm.sample_posterior_predictive(trace_eq, random_seed=rng)

az.plot_ppc(ppc_eq)
```

![PPC](/docs/assets/images/exponential/ppc.png)

The fit looks quite good, but not yet perfect, as the observed
number of events at about 8 hours is not included in our HDI,
as well as the region close to 0, and in a true study it would be wise
to understand why this model is unable to reproduce this data.
We will see how to improve this very simple kind of model in a future post.
