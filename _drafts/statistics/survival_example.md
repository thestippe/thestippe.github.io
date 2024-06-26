---
layout: post
title: "Application of survival analysis 1"
categories: /statistics/
subcategory: "Advanced models"
tags: /survival_continuous/
date: "2024-02-02"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Survival analysis with continouous time"
section: 3
---

In the previous post we introduced survival
analysis and we discussed how to correctly treat
censorship.
In this post we will see an application of survival analysis.

## The study
In this post we will use the "E1684" melanoma dataset available in the SurvSet python package.

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt
from SurvSet.data import SurvLoader

rng = np.random.default_rng()
loader = SurvLoader()

df_melanoma, ref_melanoma = loader.load_dataset(ds_name = 'e1684').values()

# Dataset reference

ref_melanoma
```

<div class='code'>
<a href='https://www.rdocumentation.org/packages/smcure/versions/2.0/topics/e1684'>https://www.rdocumentation.org/packages/smcure/versions/2.0/topics/e1684</a>
</div>

```python
df_melanoma.head()
```

|    |   pid |   event |    time |   num_age | fac_sex   | fac_trt   |
|---:|------:|--------:|--------:|----------:|:----------|:----------|
|  0 |     0 |       1 | 1.15068 | -11.0359  | M         | IFN       |
|  1 |     1 |       1 | 0.62466 |  -5.12904 | M         | IFN       |
|  2 |     2 |       0 | 1.89863 |  23.186   | F         | Control   |
|  3 |     3 |       1 | 0.45479 |  11.1449  | F         | Control   |
|  4 |     4 |       1 | 2.09041 | -13.3208  | M         | Control   |

There variable "time" is our regression variable, the "event" column indicates if
the event happened, and its value is 1 if the event happened (the patient died)
while it is 0 if the event is censored.

```python
len(df_melanoma)
```

<div class='code'>
284
</div>

Let us count how many entries are censored

```python
(df_melanoma['event']==0).astype(int).sum()
```

<div class='code'>
88
</div>

```python
df_melanoma['time'].max()
```
<div class='code'>
9.64384
</div>

## The model

We will use a Weibull likelihood, which is a quite flexible distribution,
which allows for fat tails, and it should thereby be more robust than
a Gamma distribution.

$$
Y \sim \mathcal{Weibull}(\sigma, \lambda)
$$

The Weibull distribution has pdf

$$
p(x | \alpha, \beta) = \alpha \frac{x^{\alpha-1}}{\beta^\alpha} e^{-(x/\beta)^\alpha}
$$

Both the parameters must be positive, and the mean of the distribution is

$$
\mu = \beta \Gamma\left(1+\alpha^{-1}\right)
$$

We want to assess the effectiveness of the treatment. The first possibility
is to fit two different models, one per treatment. Another very common
possibility is to use the treatment as a regression variable, and we will
use this method.
We define the covariate 

$$
x =
\begin{cases}
1 & treatment=IFN\\
0 & treatment=Control\\
\end{cases}
$$

and we assume

$$
\begin{align}
&
\lambda = \exp\left(\beta_0 + \beta_1 x \right)
\\
&
\sigma = \exp\left(\alpha \right)
\\
&
\alpha, \beta_i \sim \mathcal{N}(0, 100)
\\
\end{align}
$$

In this way we ensure that both the parameters are positive,
and the priors are very uninformative.
Let us introduce the censoring variable

```python
df_melanoma['censoring'] = [None if x==1 else y for x, y in zip(df_melanoma['event'], df_melanoma['time'])]
```

```python
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=100)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=2)
    lam = pm.math.exp(beta[0] + beta[1]*df_melanoma['trt'])
    dist = pm.Weibull.dist(alpha=pm.math.exp(alpha), beta=lam)
    y = pm.Censored('y', dist, lower=None, upper=df_melanoma['censoring'], observed=df_melanoma['time'])
    
    trace = pm.sample(draws=5000, chains=4, tune=5000, random_seed=rng)

az.plot_trace(trace)
```

![The trace of our model](/docs/assets/images/statistics/survival_melanoma/trace.webp)

## Treatment comparison

By the above figure we observe that $\beta_1>0\,,$
and this indicates that the test treatment is more effective than the control one.
This becomes clearer by showing the distribution of the mean $\mu$ 

``` python

mu0 = np.exp(trace.posterior['beta'].values.reshape(-1, 2)[:, 0])*np.exp(gammaln(1+1/np.exp(trace.posterior['alpha'].values.reshape(-1))))
mu1 = np.exp(trace.posterior['beta'].values.reshape(-1, 2)[:, 0]+trace.posterior['beta'].values.reshape(-1, 2)[:, 1])*np.exp(gammaln(1+1/np.exp(trace.posterior['alpha'].values.reshape(-1))))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(mu0, density=True, bins=np.arange(0, 15, 0.2), alpha=0.6, label='Control')
ax.hist(mu1, density=True, bins=np.arange(0, 15, 0.2), alpha=0.6, label='IFN')
ax.set_xlim([0, 15])
ax.set_ylim([0, 0.5])
legend = plt.legend()
fig.tight_layout()
```

![The posterior for the parameter mu](/docs/assets/images/statistics/survival_melanoma/mean.webp)

The mean for the test treatment is typically higher for the test group
than for the control group, and the peak of the mean for the IFN
treatment is roughly twice than the one for the control treatment.

Let us also take a look at the survival function, which is simply

$$
S(t, \alpha, \beta) = e^{-(t/\beta)^\alpha}
$$

```python
def S(t, alpha, beta):
    y = (t/beta)**alpha
    return np.exp(-y)

t_pl =  np.arange(0., 40, 0.02)
s0 = [np.mean(S(t, alph, b0)) for t in t_pl]

alph = np.exp(trace.posterior['alpha'].values.reshape(-1))
b0 = np.exp(trace.posterior['beta'].values.reshape(-1,2)[:, 0])
b1 = np.exp(trace.posterior['beta'].values.reshape(-1,2)[:, 0]+trace.posterior['beta'].values.reshape(-1,2)[:, 1])

s0 = [np.mean(S(t, alph, b0)) for t in t_pl]
s0_low = [np.quantile(S(t, alph, b0), q=0.025) for t in t_pl]
s0_high = [np.quantile(S(t, alph, b0), q=0.975) for t in t_pl]

s1 = [np.mean(S(t, alph, b1)) for t in t_pl]
s1_low = [np.quantile(S(t, alph, b1), q=0.025) for t in t_pl]
s1_high = [np.quantile(S(t, alph, b1), q=0.975) for t in t_pl]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t_pl, s0, label='Control')
ax.fill_between(t_pl, s0_low, s0_high, alpha=0.5, color='lightgray')

ax.plot(t_pl, s1, label='IFN')
ax.fill_between(t_pl, s1_low, s1_high, alpha=0.5, color='green')
ax.set_ylim([0, 1])
ax.set_xlim([0, t_pl[-1]])

ax.set_title(f'S(t)')
legend = plt.legend(frameon=False)
```

![The survival functions](/docs/assets/images/statistics/survival_melanoma/survival.webp)

We can safely conclude that, for the patients in this study, the IFN
treatment gives better results than the control one.

## Conclusions

We discussed an application of survival analysis with continuous time, we explained how
to include the regressor dependence in bayesian survival analysis
and we also introduced the Weibull distribution.

## Suggested readings

- <cite>Ibrahim, J. G., Chen, M., Sinha, D. (2013). Bayesian Survival Analysis. Switzerland: Springer New York.</cite>
