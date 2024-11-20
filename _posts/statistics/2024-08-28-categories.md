---
layout: post
title: "Multidimensional distributions"
categories: /statistics/
tags: /multidimensional/
subcategory: "Simple models"
date: "2024-08-28"
section: 4
# image: "/docs/assets/images/perception/eye.jpg"
description: "Dealing with more than two categories"
---

We will now discuss some multidimensional generalization of the previously introduced
distributions.

## The multinomial model

The categorical model can be seen as a generalization of the Bernoulli model,
and if the binomial model is the sum of $n$ Bernoulli trials,
the multinomial model is the sum of $n$ categorical trials.

<details class="math-details">
<summary> The categorical distribution
</summary>

The categorical distribution is the most general distribution
over the set of $k$ distinct elements, and it is defined as

$$
p(x | \theta_1,\dots \theta_k) = \theta_i \, if \, x = i
$$
where $$x \in \{1,2,\dots,k \}\,.$$

Since the total probability must be one, we have that

$$\sum_{i=0}^k \theta_i = 1 $$

</details>

As an example, let us consider the 2022 Formula One championship, and
let us see what's the winning probability of the best pilots.
The dataset can be found [here](https://github.com/toUpperCase78/formula1-datasets/blob/master/Formula1_2022season_raceResults.csv),
and the results for the wins are

|    | Driver          |   N. win|
|---:|:----------------|--------:|
|  0 | Carlos Sainz    |       1 |
|  1 | Charles Leclerc |       3 |
|  2 | George Russell  |       1 |
|  3 | Max Verstappen  |      15 |
|  4 | Sergio Perez    |       2 |

As a prior we will take a **Dirichlet** distribution,
which is the generalization of the beta distribution to $n$ elements.
Its support is the n-dimensional unit simplex

$$
\begin{align}
& x = (x_1,\dots, x_n) \\
& 0 \leq x_i \leq 1 \\
& \sum_{i=1}^n x_i = 1
\end{align}
$$

The Dirichlet takes an $n$ dimensional vector of real positive elements $\alpha = (\alpha_1,\dots,\alpha_n)$,
and we will take $\alpha_1=\dots=\alpha_n = \frac{1}{n}\,.$

Let us now estimate the winning probability for each of them

```python
import pandas as pd
import arviz as az
import pymc as pm
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

df_f1 = pd.read_csv(
    'https://raw.githubusercontent.com/toUpperCase78/formula1-datasets/master/Formula1_2022season_raceResults.csv')

df_red = df_f1[df_f1['Position']=='1'].groupby('Driver').count()


seed = np.random.default_rng(seed=42)

```

In order to build our model, we must map each pilot to an integer, and this can be done
with pandas' **factorize** function. 

```python
factors =  pd.factorize(df_f1[df_f1['Position']=='1']['Driver'])
y_obs = factors[0]
```

Now we can run our model

```python
with pm.Model() as cat_model:
    p = pm.Dirichlet('p', a=np.ones(len(df_red))/len(df_red))
    y = pm.Categorical('y', p=p, observed=y_obs)
    trace = pm.sample(nuts_sampler='numpyro', random_seed=seed)

az.plot_trace(trace)
```

![The trace for the multinomial model](/docs/assets/images/statistics/categories/trace.webp)

Let us take a better look at our estimates.
First of all, we will build a dataframe to match each factor to the corresponding
name.

```python
df_names = pd.DataFrame.from_dict({'name': factors[1].values,
                             'number': range(len(factors[1].values))})

```

|    | name            |   number |
|---:|:----------------|---------:|
|  0 | Charles Leclerc |        0 |
|  1 | Max Verstappen  |        1 |
|  2 | Sergio Perez    |        2 |
|  3 | Carlos Sainz    |        3 |
|  4 | George Russell  |        4 |

We can now make our forest plot, by keeping in mind that the $y$ axis goes from the
bottom to the top (we must therefore revert the order of our dataframe).

```python
fig = plt.figure()
ax = fig.add_subplot(111)
az.plot_forest(trace, ax=ax)
ax.set_yticklabels([f"{elem}" for elem in df_names['name'][::-1]])
fig.tight_layout()
```

![The forest plot for the probabilities](/docs/assets/images/statistics/categories/forest.webp)


The components are strongly correlated

```python
az.plot_pair(trace, kind='kde')
```

![The pair plot of the probabilities](/docs/assets/images/statistics/categories/kde.webp)

We can now take a look at the posterior predictive check

```python
with cat_model:
    ppc = pm.sample_posterior_predictive(trace)
```

Let us not plot the posterior predictive

```python
fig = plt.figure()
ax = fig.add_subplot(111)
az.plot_ppc(ppc, ax=ax)
ax.set_xticks(df_names['number']+0.5)
ax.set_xlim([0, 1+np.max(df_names['number'])])
ax.set_xticklabels(df_names['name'], rotation=45, fontsize=11)
```

![The ppc for the multinomial model](/docs/assets/images/statistics/categories/ppc_categorical.webp)

As you can see, our model accurately reproduces the observed data.

## Multivariate normal

Since the normal distribution with zero mean only depends on $x$ via $x^2/\sigma^2\,,$
we can immediately generalize it by replacing $x^2/\sigma^2$ with any
positively-defined form $x \cdot \Sigma \cdot x\,,$
and the generalization to the general $\mu$ case is straightforward.

A less straightforward choice is how to provide a prior for $\Sigma\,,$
and the most common solution is to use the [LKJ distributed correlations
](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html).

We will use this model to verify if there is any association between the speed 
and the alcohol consumption in the car crash dataset, which comes with the seaborn library.

```python
df_car = sns.load_dataset('car_crashes')

sns.pairplot(df_car[['speeding', 'alcohol']])
```

![The pairplot of the relevant variables](/docs/assets/images/statistics/categories/car_crash.webp)

As we can see, we cannot treat them independently, since the expected value
for one variable depends on the other variable.
As an example, if we fix "alcohol" to 5 we get an expected value for speed
which is different from the expected value that we would get by fixing alcohol
to 7.5.
The simplest way to implement this behavior for real data is by means of the multivariate
normal model.

```python
with pm.Model() as mvnorm:
    mu = pm.Normal('mu', mu=0, sigma=20, shape=(2))
    sd_dist = pm.HalfNormal.dist(2.0, size=2)
    chol, corr, sigmas = pm.LKJCholeskyCov('sigma', eta=1., n=2, sd_dist=sd_dist)
    y = pm.MvNormal('y', mu=mu, chol=chol, observed=df_car[['speeding', 'alcohol']])
    trace_car = pm.sample(nuts_sampler='numpyro', random_seed=seed)

az.plot_trace(trace_car, 
                       coords={"sigma_corr_dim_0": 0, "sigma_corr_dim_1": 1})
```

![The trace plot of the multivariate normal model](/docs/assets/images/statistics/categories/trace_car.webp)

The trace looks good. We included the coordinates option because the diagonal
terms in the correlation matrix are, by construction, always one.
This causes some issue to arviz that, when plotting, assumes that what has been provided
to the plot function, is a random variable with more than one value.
We can now turn to the posterior predictive checks.
We will do this as follows

```python
with mvnorm:
    y_p = pm.MvNormal('y_p', mu=mu, chol=chol)
    ppc_car = pm.sample_posterior_predictive(trace_car, var_names=['y', 'y_p'])

fig = plt.figure()
ax = fig.add_subplot()
az.plot_pair(ppc_car, var_names=['y_p'], kind='kde', group='posterior_predictive', ax=ax)
ax.scatter(x=df_car['speeding'], y=df_car['alcohol'], color='lightgray')
fig.tight_layout()
```

![The PPC for the multivariate normal model](/docs/assets/images/statistics/categories/ppc_car.webp)

The PPC looks quite good too, except for few outliers. These should be carefully investigated
by considering a more robust model or by changing the model structure and including additional
covariates. Since this goes beyond the scope of this post, however, we will leave the reader deal with
this problem.

We can now inspect the posterior density. Since we expect the correlation
between the variables to be relevant, we will only show the 2d joint kernel density estimate
of the posterior.

```python
az.plot_pair(trace_car,
             var_names=['mu', 'sigma', 'sigma_corr'],
            kind='kde',
                       coords={"sigma_corr_dim_0": 0, "sigma_corr_dim_1": 1})
```

![The pair plot of the posterior distribution](/docs/assets/images/statistics/categories/kde_car.webp)

## Conclusions

As we have seen, for the normal and for the binomial model,
it straightforward to immediately generalize the one dimensional model
to the multidimensional one.
This is not true for all the models, but often the multivariate normal
and the multinomial models are good starting points to build more
involved models.


```python
%load_ext watermark
```
```python
%watermark -n -u -v -iv -w -p xarray,pytensor,numpyro,jax,jaxlib
```

<div class="code">
Last updated: Wed Nov 20 2024
<br>

<br>
Python implementation: CPython
<br>
Python version       : 3.12.7
<br>
IPython version      : 8.24.0
<br>

<br>
xarray  : 2024.9.0
<br>
pytensor: 2.25.5
<br>
numpyro : 0.15.0
<br>
jax     : 0.4.28
<br>
jaxlib  : 0.4.28
<br>

<br>
seaborn   : 0.13.2
<br>
pymc      : 5.17.0
<br>
numpy     : 1.26.4
<br>
matplotlib: 3.9.2
<br>
pandas    : 2.2.3
<br>
arviz     : 0.20.0
<br>

<br>
Watermark: 2.4.3
</div>