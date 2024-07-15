---
layout: post
title: "Random models and mixed models"
categories: /statistics/
subcategory: "Hierarchical models"
tags: /random_models_intro/
date: "2024-04-28"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Making inference on subgroups"
section: 5
---

<!--TODO: write model using the coords construct! Find better syntax! include elpd estimate!-->


There are many situations where you want to understand the relation between two
variables at the subgroup level rather than at the level of the entire sample.
Random effect models are linear models where each subgroup has its own
slope and intercept, while in mixed effect models you either assign
a slope to each subgroup and a unique intercept or vice-versa.

![](/docs/assets/images/statistics/random_models/models.webp)

Let us see what are the differences between a random effect model and a fixed
effect model by looking at the data of [this study](https://www.key2stats.com/data-set/view/1040), where the authors restricted the number of hours of
a set of participants for ten days and analyzed the reaction time of the participants
to a test. The dataset only includes the set of participants who only slept
for three hours per night.

```python
import pandas as pd
import numpy as np
import jax.numpy as jnp
import seaborn as sns
from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
import pymc.sampling_jax as pmj

df = pd.read_csv('data/Reaction.csv')

df.head()
```

|    |   Unnamed: 0 |   X |   Reaction |   Days |   Subject |
|---:|-------------:|----:|-----------:|-------:|----------:|
|  0 |            1 |   1 |    249.56  |      0 |       308 |
|  1 |            2 |   2 |    258.705 |      1 |       308 |
|  2 |            3 |   3 |    250.801 |      2 |       308 |
|  3 |            4 |   4 |    321.44  |      3 |       308 |
|  4 |            5 |   5 |    356.852 |      4 |       308 |

Let us normalize the data before analyzing them,
and let us assume

$$
y_i = \alpha + \beta X_{i} + \varepsilon_i
$$

```python
df['y'] = (df['Reaction'] - mean)/df['Reaction'].std()

with pm.Model() as model_fixed:
    alpha = pm.Normal('alpha', mu=0, sigma=500)
    beta = pm.Normal('beta', mu=0, sigma=500)
    sigma = pm.HalfNormal('sigma', sigma=500)
    yhat = pm.Normal('y', mu=alpha + beta*df['Days'], sigma=sigma, observed=df['y'])
    trace_fixed = pm.sample()

az.plot_trace(trace_fixed)
```

![](/docs/assets/images/statistics/random_models/trace_fixed.webp)

```python
yfit = jnp.outer(trace_fixed.posterior['alpha'].values.reshape(-1), jnp.ones(len(df['Days'].drop_duplicates().values)))+jnp.outer(trace_fixed.posterior['beta'].values.reshape(-1), jnp.arange(len(df['Days'].drop_duplicates().values)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(df['Days'].drop_duplicates().values, jnp.quantile(yfit, q=0.025, axis=0),jnp.quantile(yfit, q=0.975, axis=0),
               color='lightgray', alpha=0.8)
ax.plot(df['Days'].drop_duplicates().values, jnp.mean(yfit, axis=0))
sns.scatterplot(df, x='Days', y='y', hue='Subject', ax=ax)
ax.get_legend().remove()

```

![](/docs/assets/images/statistics/random_models/ppc_fixed.webp)

The estimated trend agrees with the observed one, but our model
fails to reproduce the observed variability despite the large prior variance.
Let us now compare the previous model with a random effect model.
In this model we will assume that each individual has his/her own intercept,
which represents the response without sleeping restrictions, as well as 
his/her own slope, which represents the average change in performance
after one day of sleep restrictions.
We will fully leverage bayesian statistics, and assume that the parameters
have a common prior. Another possible choice would have been to assume that
each of them had a separate prior, but in this way we can share information across the individuals. We will also assume that the slope and the intercept
are correlated.

$$
\begin{align}
y_i &= \alpha_{[j]i} + \beta_{[j]i} X_{i} + \varepsilon_i
\\
\varepsilon_i & \sim \mathcal{N}(0, \sigma)
\\
\begin{pmatrix}
\alpha_{[j]i} \\
\beta_{[j]i}
\end{pmatrix}
& \sim \mathcal{N}(\mu, \Sigma)
\\
\mu_i & \sim \mathcal{N}(0, 10)
\\
\Sigma & \sim \mathcal{LKJ}(1)
\end{align}
$$

```python
with pm.Model() as model:
    mu = pm.Normal('mu', sigma=10, shape=(2))
    sd_dist = pm.HalfNormal.dist(sigma=5, size=2)
    chol, corr, sig = pm.LKJCholeskyCov('sig', n=2, eta=1.0, sd_dist=sd_dist)
    sigma = pm.HalfNormal('sigma', sigma=5)
    alpha = pm.MvNormal(f'alpha', mu=mu, chol=chol, shape=(len(df['Subject'].drop_duplicates()), 2))
    alpha_new = pm.MvNormal(f'alpha_new', mu=mu, chol=chol, shape=(2))
    yhat_new = pm.Normal(f'yhat_new', mu=alpha_new[0]+alpha_new[1]*np.arange(10), sigma=sigma)
    for k, elem in enumerate(df['Subject'].drop_duplicates()):
        df_red = df[df['Subject']==elem]
        yhat = pm.Normal(f'yhat_{k}', mu=alpha[k][0]+alpha[k][1]*df_red['Days'], sigma=sigma, observed=df_red['y'])
```

![](/docs/assets/images/statistics/random_models/trace.webp)

In the above model, we already implemented the probability distribution
for a new hypothetical participant $\hat{y}_{new}$.

```python
with model:
    ppc = pm.sample_posterior_predictive(trace)

x_pl = np.arange(10)
fig, ax=plt.subplots(nrows=6, ncols=3, figsize=(9, 8))
for i in range(6):
    for j in range(3):
        k =3*i + j
        df_red = df[df['Subject']==sub_dict_inv[k]]
        y_pl = ppc.posterior_predictive[f'yhat_{k}'].mean(dim=['draw', 'chain'])
        y_m = ppc.posterior_predictive[f'yhat_{k}'].quantile(q=0.025, dim=['draw', 'chain'])
        y_M = ppc.posterior_predictive[f'yhat_{k}'].quantile(q=0.975, dim=['draw', 'chain'])
        ax[i][j].fill_between(x_pl, y_m, y_M, alpha=0.8, color='lightgray')
        ax[i][j].plot(x_pl, y_pl)
        ax[i][j].scatter(df_red['Days'], df_red['y'])
        ax[i][j].set_ylim(-4, 4)
        ax[i][j].set_title(f"i={k}")
        ax[i][j].set_yticks([-4, 0, 4])
fig.tight_layout()
```

![](/docs/assets/images/statistics/random_models/ppc_pooled.webp)

The performances of the new model are way better than the previous one,
and this is not surprising since we have many more parameters.

We can now verify how much does the average performance degradation changes
with the participant.

```python
az.plot_forest(trace, var_names=['alpha'], coords={'alpha_dim_1':1})
```

![](/docs/assets/images/statistics/random_models/forest_pooled.webp)

We can also predict how will a new participant perform

```python
fig = plt.figure()
ax = fig.add_subplot(111)

y_pl = trace.posterior[f'yhat_new'].mean(dim=['draw', 'chain'])
y_m = trace.posterior[f'yhat_new'].quantile(q=0.025, dim=['draw', 'chain'])
y_M = trace.posterior[f'yhat_new'].quantile(q=0.975, dim=['draw', 'chain'])
ax.plot(np.arange(10), y_pl)
ax.fill_between(np.arange(10), y_m, y_M, color='lightgray', alpha=0.8)

```

![](/docs/assets/images/statistics/random_models/pp_new.webp)

Another advantage of the hierarchical model is that we can estimate
the distribution of the slope and intercept for a new participant

```python
az.plot_pair(trace, group='posterior', var_names=['alpha_new'], kind='kde')
```

![](/docs/assets/images/statistics/random_models/alpha_new.webp)

## Conclusions

We discussed the main features of random models, and we discussed when 
it may be appropriate to use them.
We have also seen what are the advantages of implementing a hierarchical
structure on random effect models.
