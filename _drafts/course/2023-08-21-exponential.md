---
layout: post
title: "The exponential model"
categories: course/intro/
tags: /exponential/
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

|    |   #EventID | Time                       |   Latitude |   Longitude |   Depth/Km | Author               |   Catalog |   Contributor |   ContributorID | MagType   |   Magnitude | MagAuthor   | EventLocationName                                | EventType   |
|---:|-----------:|:---------------------------|-----------:|------------:|-----------:|:---------------------|----------:|--------------:|----------------:|:----------|------------:|:------------|:-------------------------------------------------|:------------|
|  0 |   35812591 | 2023-08-13 00:08:52.680000 |    40.4975 |     16.173  |       33.4 | SURVEY-INGV          |       nan |           nan |             nan | ML        |         1.3 | --          | 1 km E Accettura (MT)                            | earthquake  |
|  1 |   35812671 | 2023-08-13 00:44:32.400000 |    37.8388 |     15.5168 |       10.3 | SURVEY-INGV          |       nan |           nan |             nan | ML        |         1.3 | --          | Stretto di Messina (Reggio di Calabria, Messina) | earthquake  |
|  2 |   35812691 | 2023-08-13 01:00:51.650000 |    46.6427 |      8.3947 |       10.8 | SURVEY-INGV          |       nan |           nan |             nan | ML        |         1.7 | --          | Svizzera (SVIZZERA)                              | earthquake  |
|  3 |   35812831 | 2023-08-13 01:50:30.440000 |    43.013  |     13.0773 |        7.6 | SURVEY-INGV          |       nan |           nan |             nan | ML        |         0.7 | --          | 3 km SW Fiordimonte (MC)                         | earthquake  |
|  4 |   35812921 | 2023-08-13 02:17:17.329000 |    40.8217 |     14.1372 |        0.9 | SURVEY-INGV-OV#SiSmi |       nan |           nan |             nan | Md        |         1.1 | --          | Campi Flegrei                                    | earthquake  |

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
    trace_eq = pmjax.sample_numpyro_nuts(draws=2000, tune=2000, chains=4,
                      return_inferencedata=True, random_seed=rng)

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
