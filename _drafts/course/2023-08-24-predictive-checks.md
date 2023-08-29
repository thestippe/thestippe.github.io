---
layout: post
title: "Predictive checks"
categories: course/intro/
tags: /predictive-checks/
---

In the previous post we saw some methods which allows
us to spot problems in the trace evaluation.
In this post we will look at some methods to check if our model is able to correctly
reproduce the relevant features of the data.
We will look at the "wing length" dataset, which is a quite well known dataset,
representing the length of 100 houseflies, expressed in units of $10^{-1}$ mm.

This dataset is well known, as it represents an excellent example of normally distributed
real world data.

Let us first of all load the libraries we will use and the dataset

```python
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt 
from pytensor.tensor.math import gammaln
from sklearn.datasets import load_iris
from itables import init_notebook_mode

rng = np.random.default_rng(1)
init_notebook_mode(all_interactive=True)

plt.style.use("seaborn-v0_8-darkgrid")

df_length = pd.read_csv('https://seattlecentral.edu/qelp/sets/057/s057.txt',
                        header=None).rename(columns={0: 'wing_length'})
```

Before looking at the data, let us think about the problem we are trying do model.
What could be a typical length scale for the wing length of a housefly?
A housefly is as long as the nail of a kid, so somewhere between few millimeters
and a centimeter.
We can thus use our model, which spans all the range $0-100$ (remember we are working
in units of $10^{-1}$ mm)

$$
\begin{align}
& y \sim Normal(\mu, \sigma) \\
& \mu \sim Normal(0, 50) \\
& \sigma \sim HalfCauchy(0, 50)
\end{align}
$$

Imagine a friend of yours tells you he is sure that a reasonable length scale
is around one millimeter with an uncertainty of the order of $0.1$ mm.
He will use the following model:

$$
\begin{align}
& y \sim Normal(\mu, \sigma) \\
& \mu \sim Normal(10, 1) \\
& \sigma \sim Exponential(0.5)
\end{align}
$$

So you decide to bet, and the model which will be more accurate in describing the data
will win.

His model is implemented as

```python
with pm.Model() as model_0:
    mu = pm.Normal('mu', mu=10, sigma=1)
    sigma = pm.Exponential('sigma', lam=0.5)
    p = pm.Normal("y", mu=mu, sigma=sigma, observed=df_length ['wing_length'])
```

He can now take a look at the prior predictive check

```python
with model_0:
    prp0 = pm.sample_prior_predictive()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(prp0.prior_predictive['y'].values.reshape(-1), density=True, bins=100, alpha=0.5)
```


![Prior predictive friend](/docs/assets/images/predictive_checks/prior_predictive_0.jpg)

The prior predictive is good for him, but you think it is too restrictive, as you
are not sure about the informations you have, and you know he is not a
true expert in this field.
You can now implement your model:

```python
with pm.Model() as model_1:
    mu = pm.Normal('mu', mu=0, sigma=50)
    sigma = pm.HalfCauchy('sigma', beta=50)
    p = pm.Normal("y", mu=mu, sigma=sigma, observed=df_length ['wing_length'])
```

And you now check that your prior predictive spans the entire range $0-100$

```python
with model_1:
    prp1 = pm.sample_prior_predictive()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(prp1.prior_predictive['y'].values.reshape(-1), density=True, bins=np.arange(-100, 100, 2), alpha=0.5)
ax.set_xlim([-100, 100])
```

![Prior predictive ours](/docs/assets/images/predictive_checks/prior_predictive_1.jpg)

The sample covers the entire range $[0-100]\,,$ so it looks good for you.
You can now both run your models

```python
with model_0:
    tr0 = pm.sample()

with model_1:
    tr1 = pm.sample()
```

And verify your traces

```python
az.plot_trace(tr0)
```

![Trace friend](/docs/assets/images/predictive_checks/trace_housefly_pp_0.jpg)

```python
az.plot_trace(tr1)
```

![Trace ours](/docs/assets/images/predictive_checks/trace_housefly_pp_1.jpg)

Your trace looks fine, but your friend looks a little bit worried,
as $\sigma$ is very large with respect to what he expected.

You can finally verify which model does a better job in reproducing the data:

```python
with model_0:
    pp0 = pm.sample_posterior_predictive(tr0)

az.plot_posterior_predictive(pp0)
```

![PP friend](/docs/assets/images/predictive_checks/posterior_predictive_housefly_pp_0.jpg)

```python
with model_1:
    pp1 = pm.sample_posterior_predictive(tr1)

az.plot_posterior_predictive(pp1)
```

![PP ours](/docs/assets/images/predictive_checks/posterior_predictive_housefly_pp_1.jpg)

While our mean estimate corresponds to the data within a good accuracy,
his sampled posterior predictive lies far away from the data.

## Conclusions and take-home message
- Always perform the prior predictive check to ensure that the data, as you guess they are located, is not unlikely.
- You don't have to exactly reproduce the data, as this would led you your model to overfit.
- If your problem is too complex for a simple prior predictive check, try and sample them and run your model with fake data.
- If you have no clue about the data distribution, you can perform your prior predictive with a small portion of the data (but you should then exclude those data from the analysis, as you should never use twice your data).
- Always make sure that your model is able to accurately reproduce the data with the posterior predictive check.
- In complex problem, the model will unlikely be able to _exactly_ reproduce your data, but you should at least make sure that you can reproduce the _relevant features_ of your data. You should also be sure and understand what your model fails to reproduce, and possibly why.
