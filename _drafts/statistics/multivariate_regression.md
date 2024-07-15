---
layout: post
title: "Multivariate regression"
categories: /statistics/
subcategory: "Regression"
tags: /linear_regression/
date: "2024-03-17"
section: 0
# image: "/docs/assets/images/perception/eye.jpg"
description: "Including many covariates"
---

When dealing with real-world datasets, you will rarely only
have to deal with only one independent variable.
Here we will see how to adapt our framework to this case.
As you will see, doing so is straightforward, at least in theory.
In practice, this is not true, as deciding how to improve your model may be a tricky question,
and only practice and domain knowledge will help you in this task.

## The multi-linear model

We will use the bmd dataset from [this repo](https://raw.githubusercontent.com/Sutanoy/Public-Regression-Datasets/main/bmd.csv).
Here we want to analyze how two test medications affect the bone mass density (bmd).

```python
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pymc as pm
import arviz as az

df = pd.read_csv('https://raw.githubusercontent.com/Sutanoy/Public-Regression-Datasets/main/bmd.csv')

df.head()
```

|    |    id |     age | sex   | fracture    |   weight_kg |   height_cm | medication     |   waiting_time |    bmd |
|---:|------:|--------:|:------|:------------|------------:|------------:|:---------------|---------------:|-------:|
|  0 |   469 | 57.0528 | F     | no fracture |          64 |       155.5 | Anticonvulsant |             18 | 0.8793 |
|  1 |  8724 | 75.7412 | F     | no fracture |          78 |       162   | No medication  |             56 | 0.7946 |
|  2 |  6736 | 70.7789 | M     | no fracture |          73 |       170.5 | No medication  |             10 | 0.9067 |
|  3 | 24180 | 78.2472 | F     | no fracture |          60 |       148   | No medication  |             14 | 0.7112 |
|  4 | 17072 | 54.1919 | M     | no fracture |          55 |       161   | No medication  |             20 | 0.7909 |

The outcome is continuous, and we have the following covariates:
- age (cont.)
- sex (binary)
- weight (cont.)
- height (cont.)
- medication (categorical)
- waiting time (cont.)

As you can see, "medication" may take three different values:

```python
df['medication'].unique()
```

<div class="code">
array(['Anticonvulsant', 'No medication', 'Glucocorticoids'], dtype=object)
</div>

Let us first of all convert the binary variables into a useful format

```python
df['sex'] = df['sex'].astype('category')
df['is_female'] = (df['sex']=='F').astype(int)
df['has_fracture'] = (df['fracture']=='fracture').astype(int)

sns.pairplot(df.drop(columns='id'), hue='medication')
```

![The pairplot of the dataset](/docs/assets/images/statistics/multilinear/pairplot.webp)

Let us now add two variables to handle the medication

```python
df['anticonvulsant'] = (df['medication']=='Anticonvulsant').astype(int)
df['glucocorticoids'] = (df['medication']=='Glucocorticoids').astype(int)
```

The preprocessing phase is almost done. It is now convenient to normalize the continuous
variables.

```python
for col in ['weight_kg', 'height_cm', 'waiting_time', 'age']:
    df[f"{col}_norm"] = (df[col]- df[col].mean())/(2*df[col].std())
```

Following Gelman's suggestion, we divided by two times the standard deviation.
In this way we can easily compare the posterior for continuous variables with the ones
associated to binary variables.

Let us now take a look at our model.

```python
with pm.Model() as model_0:
    alpha = pm.Normal('alpha', mu=0, sigma=5)
    beta_age = pm.Normal('beta_age', mu=0, sigma=5)
    beta_weight_kg = pm.Normal('beta_weight_kg', mu=0, sigma=5)
    beta_height_cm = pm.Normal('beta_height_cm', mu=0, sigma=5)
    beta_waiting_time = pm.Normal('beta_waiting_time', mu=0, sigma=5)
    beta_is_female = pm.Normal('beta_is_female', mu=0, sigma=5)
    beta_has_fracture = pm.Normal('beta_has_fracture', mu=0, sigma=5)
    beta_anticonvulsant = pm.Normal('beta_anticonvulsant', mu=0, sigma=5)
    beta_glucocorticoids = pm.Normal('beta_glucocorticoids', mu=0, sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=5)
    mu = alpha + beta_age*df['age_norm']+ beta_weight_kg*df['weight_kg_norm'] \
    + beta_height_cm*df['height_cm_norm']+ beta_waiting_time*df['waiting_time_norm'] + beta_is_female*df['is_female'] + beta_has_fracture*df['has_fracture'] \
    + beta_anticonvulsant*df['anticonvulsant']+ beta_glucocorticoids*df['glucocorticoids']
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=df['bmd'])
    idata = pm.sample(nuts_sampler='numpyro')
az.plot_trace(idata)
```

![](/docs/assets/images/statistics/multilinear/trace_0.webp)

The traces look fine, it is however hard to compare the distributions with such a big number
of variables. Fortunately we can use the forest plot

```python
a.plot_forest(idata, var_names='beta', filter_vars='like', combined=True)
```

![](/docs/assets/images/statistics/multilinear/forest_0.webp)

As we can see, the bone mass density is much lower among people with a fracture
than among people without a fracture,
while a higher weight is associated with a higher bmd.

Both assuming anticonvulsant and glucocorticoids may have a small negative effect,
but this effect is too small to be significant.
In any case, the sign is negative, but the error bars are compatible with 0, so
we cannot conclude anything on this.
