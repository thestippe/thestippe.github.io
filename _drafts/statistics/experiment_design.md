---
layout: post
title: "Experiment analysis"
categories: /statistics/
subcategory: "Advanced models"
tags: /experiment_analysis/
date: "2024-06-02"
# image: "/docs/assets/images/perception/eye.jpg"
description: "How to design and analyze experiment"
section: 5
---

Experimental design was developed by Fisher in the context of agriculture,
with the aim to better make decisions based on experimental data.
Nowadays experimental design is applied in many different fields,
but unfortunately there aren't many textbook which treat this
topic from a Bayesian perspective.
The advantage of a Bayesian treatment is clear if you consider
that, often, collective data requires a big effort.
As an example, running an experiment in the agriculture
may take months, so it is crucial to extract as much information as possible
from every single datum.
The reason for sticking to the traditional data analysis approaches
might be due to the fact that Fisher had a very strong positions
against the Bayesian statistics, or maybe because most of the available
software is based on Fisher's models.
It is however very easy to build Bayesian models for this kind
of application, and we will take a look at how to do so.

## Principles of experimental design
When designing an experiment, you should keep in mind three fundamental
principles:
- randomization
- replication
- blocking

When we talk about **randomization**, we refer to the random assignment of the
experimental units to different treatment condition.
Randomization allows you to reduce the systematic error, and any other residual
variation in the experimental procedure is random by construction.

**Replication** is the repetition of the experimental procedure,
from the preparation to the measurement.
The purpose of replication is to have a reliable estimate of the variance,
and without replication your effect estimate might not be reliable.

**Blocking** refers to fixing (when possible) or measuring (otherwise)
any factor which might reasonably affect the experimental outcome.
Controlling for external factors which might affect our measurement
helps us in identifying the sources of variability in the outcome,
and therefore will give us a more precise estimate of the effect.

## Completely randomized design

In a completely randomized experiment we assign the treatment to each experimental
unit completely at random.
In the simplest version of this design, we have $n \times k$ units,
and we want to randomly assign each unit to one of $n$ treatments.

As an example, consider the dataset at [this link](https://www.itl.nist.gov/div898/software/dataplot/data/BOXBLOOD.DAT).
The aim of the experiment is to determine whether different diets
had different effects on the blood coagulation time.
The example is taken from page 155 of [this textbook](https://pages.stat.wisc.edu/~yxu/Teaching/16%20spring%20Stat602/%5BGeorge_E._P._Box,_J._Stuart_Hunter,_William_G._Hu(BookZZ.org).pdf).

```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pymc as pm
import arviz as az

df = pd.read_csv('data/coagulation.csv')

fig = plt.figure()
ax = fig.add_subplot(111)
sns.boxplot(df, x='Y', hue='X1', ax=ax)
```

![The boxplot of the
coagulation dataset](/docs/assets/images/statistics/experiment_design/coagulation_boxplot.webp)

In the dataset we have 24 units, and to each unit we want to assign
one of the four possible treatments.
This could have been done by running
```python
n=4
k=6

items = list(range(n*k))
np.random.shuffle(items)
groups = [items[i*k:(i+1)*k] for i in range(n)]
```


We have different way to analyze this dataset, and we will use the following
model

$$
\begin{align}
&
y_{ij} \sim \mathcal{N}(\mu_i, \sigma)
\\
&
\sigma \sim \mathcal{HN}(0, 100)
\\
&
\mu_i \sim \mathcal{N}(0, 100)
\end{align}
$$

```python
with pm.Model() as coag_model:
    mu = pm.Normal('mu', mu=0, sigma=100, shape=(4))
    sigma = pm.HalfNormal('sigma', sigma=100)
    y = pm.Normal('y', mu=mu[df['X1']-1], sigma=sigma, observed=df['Y'])
    idata_coag = pm.sample(nuts_sampler='numpyro', draws=5000, random_seed=rng)

az.plot_trace(idata_coag)
fig = plt.gcf()
fig.tight_layout()
```

![The trace of the
coagulation model](/docs/assets/images/statistics/experiment_design/coagulation_trace.webp)

```python
az.plot_forest(idata_coag, var_names=['mu'])
```


![The forest plot of the
means for the coagulation model](/docs/assets/images/statistics/experiment_design/coagulation_forest.webp)

There is no doubt that, in this study, we observe that a diet variation
corresponds to a coagulation time variation.

## Randomized block design
In a randomized block design, we furthermore block one or more
factor (confounder) which may be a source of variability for
the outcome.
As an example, let us consider the example in chapter 4.2 of the same book.
In this experiment the authors compared the penicillin yield
of different processes (formulas).
The yield does not only depend on the process, but also on the
ingredients, so the authors used one of the main ingredients,
the corn steep liquor, as blocking factor.
The corresponding dataset can be found [here](https://www.itl.nist.gov/div898/software/dataplot/data/BOXPENIC.DAT).

In order to analyze this dataset, we will use the following model


$$
\begin{align}
&
y_{ijk} \sim \mathcal{N}(\mu_i + \tau_j, \sigma)
\\
&
\sigma \sim \mathcal{HN}(0, 100)
\\
&
\mu_i \sim \mathcal{N}(0, 100)
\\
&
\tau_j \sim \mathcal{N}(0, \rho)
\\
&
\rho \sim \mathcal{HN}(0, 100)
\end{align}
$$


```python
df_rbd = pd.read_csv('data/penicillin.csv')
```


```python
with pm.Model() as crb_model:
    
    mu = pm.Normal('mu', mu=0, sigma=100, shape=(4))
    rho = pm.HalfCauchy('rho', beta=50)
    tau = pm.Normal('tau', mu=0, sigma=rho, shape=(5))
    sigma = pm.HalfNormal('sigma', sigma=20)
    y = pm.Normal('y', mu=mu[df_rbd['T']-1]+tau[df_rbd['X']-1], sigma=sigma, observed=df_rbd['Y'])
    idata_rbd = pm.sample(nuts_sampler='numpyro', draws=5000, tune=5000, target_accept=0.99)

az.plot_trace(idata_rbd)
```

As we can clearly see, the 'X' effect has average zero, therefore
$\mu$ correctly estimates the average treatment effect.


![The trace plot of the
means for the penicillin model](/docs/assets/images/statistics/experiment_design/penicillin_trace.webp)

We can now take a better look at the values of $\mu$

```python
az.plot_forest(idata_rbd, var_names=['mu'])
```

![The forest plot of the
means for the penicillin model](/docs/assets/images/statistics/experiment_design/penicillin_forest.webp)

As you can see, the treatment 2 gives slightly higher yields with respect to the
other treatment.

When designing an experiment, the simplest solution is to use a completely randomized design,
and measure the remaining blocking variables.
This might however allocate treatment in an undesired way,
so you might decide to assign the treatment according to a probability
which depends on the blocking variable.

In fact, if a relevant variable is unbalanced across the treatment groups, you cannot
exclude that a difference into the outcome can be imputed to the different average values
of the variable across the groups.

If the blocking variable is discrete, blocking for it is  a straightforward procedure.
If it is continuous, however, in order to do so, you are forced stratify it,
namely to build a discrete variable and mapping the continuous variable to the discrete one.
As an example, when blocking on the age of a person, you might stratify it into
0-9, 10-19, 20-29 etc.
The values of the discrete variable are often named strata (plural of stratum) of levels
of the continuous variable, and the first term is generally used if you are measuring it,
while the latter is preferred when you are fixing it.
For a more in-depth discussion on this topic, you can take a look at
[https://arxiv.org/pdf/2305.18793](https://arxiv.org/pdf/2305.18793).

## Matched pairs design

The matched pairs design can be considered a special case of the randomized block design
with two treatment groups (the extension to more groups is straightforward).
Rather than randomly assigning each unit to one of two groups, we first pair
units with similar relevant features, and then we toss a coin to decide which element
of the pair belongs to which group.

This kind of pairing can be useful when we have small samples or if we have very similar
pairs of units, such as twins.

In order to illustrate the analysis method, we will re-analyze the [Orley Ashenfelter and Alan Krueger](https://www.jstor.org/stable/2117766)
article, where the authors observed the impact on the instruction on the wage on a large set
of twins.
The dataset is available [here](https://dataspace.princeton.edu/handle/88435/dsp012801pg35n).

```python
dfk = pd.read_stata('data/pubtwins.dta')

df_red = dfk[dfk['first']==1][['deduct', 'dlwage']]
```

In the above dataset, we only kept the rows corresponding to the first child,
and the columns corresponding to the number of education difference between the first
and the second child in years and their wage difference.
Since each subject has a similar test subject,
we can directly make inference on their wage difference.

```python
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(df_red['deduct'], df_red['dlwage'])
```
![](/docs/assets/images/statistics/experiment_design/twins_scatter.webp)

```python
with pm.Model() as model:
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma',  sigma=10)
    mu = beta*df_red['deduct']
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=df_red['dlwage'])
    idata_twins = pm.sample(nuts_sampler='numpyro', draws=5000, tune=5000,
                      chains=4, random_seed=rng)

az.plot_trace(idata_twins)
fig = plt.gcf()
fig.tight_layout()
```

![](/docs/assets/images/statistics/experiment_design/twins_trace.webp)

```python
xpl = np.arange(-6.5, 6.5, 0.05)
dt = idata_twins.posterior['beta'].values.reshape((-1, 1))*xpl

fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(xpl, np.quantile(dt, axis=0, q=0.03), np.quantile(dt, axis=0, q=0.97),
               color='lightgray', alpha=0.8)
ax.plot(xpl, np.mean(dt, axis=0), color='k')
ax.scatter(df_red['deduct'], df_red['dlwage'])
fig.tight_layout()
```

![](/docs/assets/images/statistics/experiment_design/twins_ppc.webp)

## Conclusions

We have discussed how to adapt some classical model used in experimental
design to make them Bayesian, and we have done so by using PyMC.
This was only an introductory discussion on the topic, as experimental design
is a very broad and active research topic.
In the next post, we will continue our discussion about
experimental design for more involved experiments.


## Suggested readings

- <cite>Box, G. E. P., Hunter, J. S., Hunter, W.G. (2005). Statistics for experimenters: design, innovation, and discovery. Wiley.</cite>
- <cite>Lawson, J. (2014). Design and Analysis of Experiments with R. CRC Press.<cite>

```python
%load_ext watermark
```


```python
%watermark -n -u -v -iv -w -p xarray,numpyro,jax,jaxlib
```

<div class="code">
Last updated: Mon Aug 19 2024
<br>

<br>
Python implementation: CPython
<br>
Python version       : 3.12.4
<br>
IPython version      : 8.24.0
<br>

<br>
xarray : 2024.5.0
<br>
numpyro: 0.15.0
<br>
jax    : 0.4.28
<br>
jaxlib : 0.4.28
<br>

<br>
matplotlib: 3.9.0
<br>
pymc      : 5.15.0
<br>
seaborn   : 0.13.2
<br>
numpy     : 1.26.4
<br>
arviz     : 0.18.0
<br>
pandas    : 2.2.2
<br>

<br>
Watermark: 2.4.3
<br>
</div>