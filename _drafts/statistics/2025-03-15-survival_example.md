---
layout: post
title: "Application of survival analysis 1"
categories: /statistics/
subcategory: "Advanced models"
tags: /survival_continuous/
date: "2025-03-15"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Survival analysis with continuous time"
section: 3
---

In the previous post we introduced survival
analysis, and we discussed how to correctly treat
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
import pytensor as pt

rng = np.random.default_rng(42)
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

## The models

We will try and fit two models:
first we will try with an Exponential likelihood,

$$
Y \sim \mathcal{Exponential}(\lambda)
$$

we will then try with a Weibull likelihood, which is a quite flexible distribution,
which allows for fatter tails, and it should thereby be more robust than
an exponential or a Gamma distribution.

$$
Y \sim \mathcal{Weibull}(\sigma, \lambda)
$$

The Weibull distribution has pdf

$$
p(x | \alpha, \lambda) = \alpha \frac{x^{\alpha-1}}{\lambda^\alpha} e^{-(x/\lambda)^\alpha}
$$

Notice that, often, the second parameter is named $\beta$,
but since we will use this method for another purpose, we will use $\lambda$
for the second parameter.
Both the parameters must be positive, and the mean of the distribution is

$$
\mu = \lambda \Gamma\left(1+\alpha^{-1}\right)
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
df_melanoma['trt'] = (df_melanoma['fac_trt']=='IFN').astype(int)
```

We will first try with the exponential model,
as discussed in chapter 2 of "Bayesian Survival Analysis"

```python
with pm.Model() as expon_model:
    beta = pm.Normal('beta', mu=0, sigma=1000, shape=2)
    lam = pm.math.exp(beta[0] + beta[1]*df_melanoma['trt'])
    dist = pm.Exponential.dist(lam=lam)
    y = pm.Censored('y', dist, lower=None, upper=df_melanoma['censoring'],
                    observed=df_melanoma['time'])
    idata_expon = pm.sample(nuts_sampler='numpyro',
                           draws=5000, random_seed=rng)

az.plot_trace(idata_expon)
fig = plt.gcf()
fig.tight_layout()
```

![The trace of the exponential model model](
/docs/assets/images/statistics/survival_melanoma/trace_expon.webp)

The image is similar, but not identical to the one of the reference.
However, our model is identical to the one provided in the reference.
In order to check this, you can try and fit the following model:

```python
with pm.Model() as expon_model_check:
    beta = pm.Normal('beta', mu=0, sigma=1000, shape=2)
    lam = pm.math.exp(beta[0] + beta[1]*df_melanoma['trt'])
    def logp(lam, nu, y):
        return nu*pm.math.log(lam)-y*lam
    y = pm.Potential('y', logp(lam, df_melanoma['event'].values, df_melanoma['time'].values))
```

The formula for the log-likelihood is the one provided in the reference.
If you try and fit the model, you will get exactly the same figure we obtained.
The reason for the different trace is, probably, that the samplers
improved quite a lot in more than 20 years, in fact our trace looks much
better than the one in the reference.

We will now improve our model and try with the Weibull model

```python
with pm.Model() as weibull_model:
    alpha = pm.Normal('alpha', mu=0, sigma=100)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=2)
    lam = pm.math.exp(beta[0] + beta[1]*df_melanoma['trt'])
    dist = pm.Weibull.dist(alpha=pm.math.exp(alpha), beta=lam)
    y = pm.Censored('y', dist, lower=None, upper=df_melanoma['censoring'],
                    observed=df_melanoma['time'])

with weibull_model:
    idata_weibull = pm.sample(nuts_sampler='numpyro', draws=5000, random_seed=rng)
```


![The trace of the Weibull model model](
/docs/assets/images/statistics/survival_melanoma/trace_weibull.webp)

As we discussed, when you have two nested model, it is generally better to
use the more general one.
We will however try and compare the models in order to see which is the more appropriate

```python
with expon_model:
    pm.compute_log_likelihood(idata_expon)

with weibull_model:
    pm.compute_log_likelihood(idata_weibull)

with expon_model:
    idata_expon.extend(pm.sample_posterior_predictive(idata_expon, random_seed=rng))

with weibull_model:
    idata_weibull.extend(pm.sample_posterior_predictive(idata_weibull, random_seed=rng))

df_compare = az.compare({'Exponential': idata_expon, 'Weibull': idata_weibull})

az.plot_compare(df_compare)
```


![The LOO model comparison](
/docs/assets/images/statistics/survival_melanoma/loo.webp)

The Weibull model seems much more appropriate to describe the data,
we will therefore stick to it from now on.


## Treatment comparison

By looking at the Weibull trace, we observe that $\beta_1>0\,,$
and this indicates that the test treatment is more effective than the control one.
This becomes clearer by showing the distribution of the mean $\mu$ 

``` python
with weibull_model:
    mu0 = pm.Deterministic('mu0', pm.math.exp(beta[0])*pm.math.exp(pt.tensor.math.gammaln(1+1/pm.math.exp(alpha))))
    mu1 = pm.Deterministic('mu1', pm.math.exp(beta[0]+beta[1])*pm.math.exp(pt.tensor.math.gammaln(1+1/pm.math.exp(alpha))))

with weibull_model:
    idata_mu = pm.sample_posterior_predictive(idata_weibull, var_names=['mu0', 'mu1'])

fig, ax = plt.subplots(nrows=2)
xlim = [2, 15]
az.plot_posterior(idata_mu, group='posterior_predictive', var_names='mu0', ax=ax[0])
az.plot_posterior(idata_mu, group='posterior_predictive', var_names='mu1', ax=ax[1])
ax[0].set_xlim(xlim)
ax[1].set_xlim(xlim)
ax[0].set_title("$\\mu_0$")
ax[1].set_title("$\\mu_1$")
fig.tight_layout()
```

![The posterior for the parameter mu](/docs/assets/images/statistics/survival_melanoma/mean_new.webp)

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

t_pl =  np.arange(0., 5, 0.02)

alph = np.exp(idata_weibull.posterior['alpha'].values.reshape(-1))
b0 = np.exp(idata_weibull.posterior['beta'].values.reshape(-1,2)[:, 0])
b1 = np.exp(idata_weibull.posterior['beta'].values.reshape(-1,2)[:, 0]+idata_weibull.posterior['beta'].values.reshape(-1,2)[:, 1])

s0 = [np.mean(S(t, alph, b0)) for t in t_pl]
s0_low = [np.quantile(S(t, alph, b0), q=0.03) for t in t_pl]
s0_high = [np.quantile(S(t, alph, b0), q=0.97) for t in t_pl]

s1 = [np.mean(S(t, alph, b1)) for t in t_pl]
s1_low = [np.quantile(S(t, alph, b1), q=0.03) for t in t_pl]
s1_high = [np.quantile(S(t, alph, b1), q=0.97) for t in t_pl]

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
to include the regressor dependence in bayesian survival analysis,
and we also introduced the Weibull distribution.

In the next post, we will discuss a more flexible way to estimate the
survival function.

## Suggested readings

- <cite>Ibrahim, J. G., Chen, M., Sinha, D. (2013). Bayesian Survival Analysis. Springer New York.</cite>


```python
%load_ext watermark
```

```python
%watermark -n -u -v -iv -w -p xarray,numpyro,jax,jaxlib
```

<div class="code">
Last updated: Sun Jul 21 2024
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
pandas     : 2.2.2
<br>
arviz      : 0.18.0
<br>
kaplanmeier: 0.2.0
<br>
pymc       : 5.15.0
<br>
pytensor   : 2.20.0
<br>
numpy      : 1.26.4
<br>
matplotlib : 3.9.0
<br>

<br>
Watermark: 2.4.3
<br>
</div>
