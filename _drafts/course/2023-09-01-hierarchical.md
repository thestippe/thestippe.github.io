---
layout: post
title: "Introduction to the hierarchical models"
categories: course/composite/
tags: /hierarchical/
---

Hierarchical models are one of the most important model families
in Bayesian statistics, and most important, it does not have an analogous
in frequentist statistics, and the reason will be clear soon [^1].

One of the possible applications of the hierarchical models is when you are dealing
with data which have a hierarchy of parameters.

As an example, think about a case where you have many districts,
for each district we have many schools and for each school we have many student,
and you want to analyze the grades of the students.

It wouldn't make much sense to assume that students coming from different
schools have the same grade distribution.

In Bayesian statistics you can use a prior for each school,
and you can also extend the hierarchy allowing for all the priors related
to schools of the same district to have a common prior, and you can
finally put a common prior do these probability distribution.

Let us see how to do this in practice by comparing hierarchical models
to the so-called pool models and no-pool models

## The Space-X launch failures
From the Space-X Wikipedia page have taken, for each Space-X mission, the number of incidents.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import pymc as pm
import arviz as az
import scipy.stats as st
from matplotlib import pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

df_spacex = pd.DataFrame({'Mission': ['Falcon 1', 'Falcon 9', 'Falcon Heavy', 'Starship'], 'N': [5, 227, 6, 1], 'y': [2, 224, 6, 0]})

df_spacex['mu'] = df_spacex['y']/df_spacex['N']

df_spacex
```

|    | Mission      |   N |   y |       mu |
|---:|:-------------|----:|----:|---------:|
|  0 | Falcon 1     |   5 |   2 | 0.4      |
|  1 | Falcon 9     | 227 | 224 | 0.986784 |
|  2 | Falcon Heavy |   6 |   6 | 1        |
|  3 | Starship     |   1 |   0 | 0        |

As you can see, the Falcon-9 and the Falcon Heavy have success rate close to 1,
while the Falcon 1 has a much lower success rate, and the Starship
has no successes at all.
We have different rocket types, each one with its own characteristics, so if we want to assess the success rate of each mission we have to assess if we want to consider them separately or together. In other words, we must decide the **pooling level**.

### No pooling
We could argue that those are different missions, and it probably doesn't make much sense
to think that the probability of success of a Starship is the same as
the one of a Falcon 9, so it is not much reasonable to put a common prior to them.
In this case, we could proceed with a no pooling model, where we consider each group separately.

```python
with pm.Model() as spacex_model_no_pooling:  
  theta = pm.Beta('theta', alpha=1, beta=1, shape=len(df_spacex['N'].values))
  y = pm.Binomial('y', p=theta, n=df_spacex['N'].values,
                  observed=df_spacex['y'].values)

pm.model_to_graphviz(spacex_model_no_pooling)
```

![The no-pooling model](/docs/assets/images/hierarchical/model_no_pooling.svg)

```python
with spacex_model_no_pooling:
  trace_spacex_no_pooling = pm.sample(2000, tune=500, chains=4,
                           return_inferencedata=True)

az.plot_trace(trace_spacex_no_pooling)
```

![The no-pooling trace](/docs/assets/images/hierarchical/trace_no_pooling.png)

```python
az.plot_forest(trace_spacex_no_pooling)
```

![Forest plot for the no-pooling model](/docs/assets/images/hierarchical/forest_no_pooling.png)

The parameter value depends on the mission, and of course in the case
of the Starship it will be heavily influenced by the prior, while
for the Falcon 9 it will be almost completely fixed by the data.
Moreover, each rocket is considered separately,
so if a new rocket is built, we wouldn't have any obvious
way to decide its success probability.

### Complete pooling
Alternatively, we could take a common prior for the four missions.
In other words, we could assume that the missions are independent identically distributed.
This is of course quite a strong assumption, and this will strongly affect
our inference.

```python
with pm.Model() as spacex_model_full_pooling:  
  theta = pm.Beta('theta', alpha=1, beta=1)
  y = pm.Binomial('y', p=theta, n=df_spacex['N'].values,
                  observed=df_spacex['y'].values)

pm.model_to_graphviz(spacex_model_full_pooling)
```

![The ful-pooling model](/docs/assets/images/hierarchical/model_full_pooling.svg)

```python
with spacex_model_full_pooling:
  trace_spacex_full_pooling = pm.sample(2000, tune=500, chains=4,
                           return_inferencedata=True)

az.plot_trace(trace_spacex_full_pooling)
```

![The full-pooling trace](/docs/assets/images/hierarchical/trace_full_pooling.png)


```python
az.plot_forest(trace_spacex_full_pooling)
```

![Forest plot for the full-pooling model](/docs/assets/images/hierarchical/forest_full_pooling.png)

In this case, of course, all the missions have the same prior,
so we would bet that a second Starship mission would almost surely have no
failures, and I am not sure I would be a good idea to do so,
as maybe the Starship has some intrinsic issue which is not shared 
with the other missions.

### Partial pooling or hierarchical models

Bayesian statistics offers us another way to proceed: we can consider the success probability separately, but instead of using numbers for the hyperparameters $a$ and $b$ we can promote them to probabilities and put a common prior to them, and this is how **hierarchical models** are built.

```python
with pm.Model() as spacex_model_hierarchical:
    alpha = pm.HalfNormal("alpha", sigma=10)
    beta = pm.HalfNormal("beta", sigma=10)
    mu = pm.Deterministic("mu", alpha/(alpha+beta))
    theta = pm.Beta('theta', alpha=alpha, beta=beta, shape=len(df_spacex['N'].values))
    y = pm.Binomial('y', p=theta, n=df_spacex['N'].values,
                  observed=df_spacex['y'].values)

pm.model_to_graphviz(spacex_model_hierarchical)
```

![The partial-pooling model](/docs/assets/images/hierarchical/model_partial_pooling.svg)

```python
with spacex_model_hierarchical:
    trace_spacex_hierarchical = pm.sample(2000, tune=500, chains=4, target_accept=0.95, return_inferencedata=True)

az.plot_trace(trace_spacex_hierarchical)
```

![The partial-pooling trace](/docs/assets/images/hierarchical/trace_partial_pooling.png)


```python
az.plot_forest(trace_spacex_hierarchical)
```

![Forest plot for the partial-pooling model](/docs/assets/images/hierarchical/forest_partial_pooling.png)


The additional parameters allow us to account for the uncertainties due to the reduced 
number of flights of the Starship and of the Falcon 1, as our no-pooled model
did.
However, there is a major advantage of this approach with respect to the no-pooled one:
we can predict the failure probability of a new rocket type.

```python
with spacex_model_hierarchical:
    theta_new = pm.Beta('theta_new', alpha=alpha, beta=beta)
    ppc_new = pm.sample_posterior_predictive(trace_spacex_hierarchical, var_names=['theta_new'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(ppc_new.posterior_predictive['theta_new'].values.reshape(-1), bins=np.arange(0, 1, 0.05), density=True)
fig = plt.gcf()
```

![Prior for a new mission](/docs/assets/images/hierarchical/theta_new.png)

We can also plot the distribution for the mean of the thetas.

```python
map_mu = st.mode(np.digitize(trace_spacex_hierarchical.posterior["mu"].values.reshape(-1), bins=np.linspace(0, 1, 100))).mode/100

fig = plt.figure()
ax = fig.add_subplot(111)
az.plot_posterior(trace_spacex_hierarchical, var_names=["mu"], ax=ax)
ax.text(0.3, 3, f'MAP: {map_mu}', fontsize=15)
```


![  ](/docs/assets/images/hierarchical/mu_plot.png)

In this case the expected success rate for a new rocket is much more careful
than the one produced by the fully pooled,
as the bad performances of some of the rocket types are properly taken into account.

## Application to meta-analysis

Bayesian hierarchical model are often used in meta-analysis and reviews,
*i.e.* in academic publications where the results of many studies are collected,
criticized and combined together.
In this kind of study using a full pooling would not be appropriate,
as each study is performed at its own conditions,
so a hierarchical model is much more appropriate to combine the results together.

We will take as an example the dataset of
[this](https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-021-06536-3)
study where the authors performed a meta-analysis to estimate the
COVID-19 mortality rate [^2].

```python
df_mortality = pd.read_csv('data/dt_mortality.csv', sep=' ')
df_mortality.head()
```

|    |    y |    N |
|---:|-----:|-----:|
|  0 |   89 |  432 |
|  1 |  219 |  828 |
|  2 |  103 |  607 |
|  3 | 1131 | 4035 |
|  4 |   75 |  565 |

Here $N$ represents the infected number, while $y$ represents the number of deceased.

```python
with pm.Model() as hierarchical_mortality:
    alpha = pm.HalfNormal("alpha", sigma=10)
    beta = pm.HalfNormal("beta", sigma=10)
    # Here we will compute both the logit of the mean and the log of the effective size of the posterior
    logit_mu = pm.Deterministic("logit_mu", np.log(alpha/beta))
    log_neff = pm.Deterministic("log_neff", np.log(alpha+beta))
    theta = pm.Beta('theta', alpha=alpha, beta=beta, shape=len(df_mortality['N'].values))
    y = pm.Binomial('y', p=theta, n=df_mortality['N'].values,
                  observed=df_mortality['y'].values)

    trace_hm = pm.sample(2000, chains=4, tune=500)

az.plot_trace(trace_hm)
```

![The trace for our model](/docs/assets/images/hierarchical/trace_meta.png)

```python
az.plot_forest(trace_hm, var_names='theta')
fig = plt.gcf()
```

Let us look at the estimated mortality rate for each study.

![The forest plot for the mortality rate](/docs/assets/images/hierarchical/forest_mortality.png)

We now plot the logit of the mean against the logarithm of the 
effective size of the posterior.


```python
az.plot_pair(trace_hm, var_names=["logit_mu", "log_neff"], kind="kde")
fig = plt.gcf()
```

![The forest plot for the mortality rate](/docs/assets/images/hierarchical/kde_mortality.png)

[^2]: Please remember that who writes has no knowledge of epidemiology, needed to assess the goodness of the analyzed studies, we will just use the dataset for illustrative purpose.).

[^1]: Nor in the likelihood school, since this framework does not allow for priors.
