---
layout: post
title: "Robust linear regression"
categories: /statistics/
subcategory: "Regression"
tags: /robust_regression/
date: "2024-01-26"
section: 1
# image: "/docs/assets/images/perception/eye.jpg"
description: "Reducing sensitivity to large deviations"
---


In some case your data may be not good enough to provide you reliable estimates with normal linear regression,
and this is the case of the conclusions drawn from
[this](https://www.cambridge.org/core/journals/american-political-science-review/article/abs/political-institutions-and-voter-turnout-in-the-industrial-democracies/D6725BBF93F2F90F03A69B0794728BF7)
article, where the author concludes that there is a significant correlation between
the voter turnout in a country and its average income inequality.
This example is a classical example of misleading result of a regression,
where the author does not provide a plot of the data, taken from
[Healy, "Data visualization, a practical introduction"](
https://www.google.it/books/edition/Data_Visualization/3XOYDwAAQBAJ?hl=it&gbpv=1&dq=Data+visualization,+a+practical+introduction&printsec=frontcover).
The data below is extracted the data from the figure of Healy's book.
South Africa corresponds to the last point.

|    |   turnout |   inequality |
|---:|----------:|-------------:|
|  0 |  0.85822  |      1.95745 |
|  1 |  0.837104 |      1.95745 |
|  2 |  0.822021 |      2.41135 |
|  3 |  0.87632  |      2.76596 |
|  4 |  0.901961 |      2.95035 |
|  5 |  0.776772 |      3.21986 |
|  6 |  0.72549  |      3.14894 |
|  7 |  0.72549  |      2.92199 |
|  8 |  0.61991  |      2.93617 |
|  9 |  0.574661 |      2.31206 |
| 10 |  0.880845 |      3.60284 |
| 11 |  0.803922 |      3.5461  |
| 12 |  0.778281 |      3.47518 |
| 13 |  0.739065 |      3.68794 |
| 14 |  0.819005 |      4.41135 |
| 15 |  0.645551 |      3.91489 |
| 16 |  0.669683 |      5.64539 |
| 17 |  0.14178  |      9.30496 |

```python
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pymc.sampling_jax as pmjax

df_turnout = pd.read_csv('data/inequality.csv')

sns.pairplot(df_turnout)
```

![The dataset pairplot](/docs/assets/images/statistics/robust_regression/inequality_pairplot.webp)

By simply plotting the data we can clearly see that there is one point,
the South Africa, which is far away from the other,
and this may have a huge impact on the fit.
Let us see this, and how one may avoid this kind of error.

## The normal linear regression

Let us start by assuming that the inequality is distributed
according to a normal linear model,
analogous to the one already discussed in the [regression post](/linear_regression).

$$
Y \sim \mathcal{N}(\mu, \sigma)
$$

where

$$
\mu = \alpha + \beta X
$$

We will assume that the precision $\tau = 1/\sigma$ is distributed according to a Half Normal
distribution. Since the inequality goes from 0 to 10, assuming a
standard deviation of $5$ for $\tau$ should be sufficient.
On the other hand, we will make the quite generous assumption that

$$\alpha \sim \mathcal{N}(0, 20)$$

$$\beta \sim \mathcal{N}(0, 20)$$

```python
with pm.Model() as model_norm:
    alpha = pm.Normal('alpha', mu=0, sigma=20)
    beta = pm.Normal('beta', mu=0, sigma=20)
    tau = pm.HalfNormal('tau', sigma=5)
    y = pm.Normal('y', mu=alpha+df_turnout['turnout'].values*beta, observed=df_turnout['inequality'].values, tau=tau)

with model_norm:
    trace_norm = pm.sample(draws=4000, chains=4, tune=4000, 
                           idata_kwargs = {'log_likelihood': True}, random_seed=rng)

az.plot_trace(trace_norm)
```

![The trace of the normal model](/docs/assets/images/statistics/robust_regression/trace_norm.webp)

The traces doesn't show any relevant issue, and for our purposes it is
sufficient this check.
Let us check our fit

```python
x_plt = np.arange(0, 1, 0.001)

with model_norm:
    y_pred = pm.Normal('y_pred', mu=alpha+x_plt*beta, tau=tau)
    
with model_norm:
    ppc_norm = pm.sample_posterior_predictive(trace_norm, var_names=['y', 'y_pred'], random_seed=rng)
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_plt, ppc_norm.posterior_predictive['y_pred'].mean(dim=['draw', 'chain']))
ax.fill_between(x_plt, ppc_norm.posterior_predictive['y_pred'].quantile(q=0.025, dim=['draw', 'chain']),
                ppc_norm.posterior_predictive['y_pred'].quantile(q=0.975, dim=['draw', 'chain']), alpha=0.5, color='grey')
ax.scatter(df_turnout['turnout'].values, df_turnout['inequality'].values)
```

![The posterior preditcive distribution of our model](/docs/assets/images/statistics/robust_regression/ppc_norm.webp)

The error bands correctly reproduce almost all the data. However,
since the South Africa is far away from the other countries,
it may happen that its behavior strongly influences the fit.

Let us now use a more robust model.
In order to make it more robust, which in this context means
less sensitive to isolated data, let us take a t-Student likelihood
instead of a normal one.

We will leave the parameters $\alpha\,, \beta$ and $\tau = \frac{1}{\sigma}$
unchanged, but we must choose a prior for the number of degrees of
freedom $\nu\,.$

We wand a robust estimate, so we want a prior with a small
number of degrees of freedom. However, $\nu \approx 0$
can be hard to handle from a numeric perspective,
since the resulting distribution decreases very slowly 
as one steps away from the peak.
For the above reason, we choose a Gamma prior with $\alpha=4$
and $\beta=2\,.$

```python
with pm.Model() as model_robust:
    alpha = pm.Normal('alpha', mu=0, sigma=20)
    beta = pm.Normal('beta', mu=0, sigma=20)
    tau = pm.HalfNormal('tau', sigma=5)
    nu = pm.Gamma('nu', alpha=4, beta=2)
    y = pm.StudentT('y', mu=alpha+df_turnout['turnout'].values*beta, observed=df_turnout['inequality'].values, sigma=1/tau, nu=nu)

with model_robust:
    trace_robust = pm.sample(draws=4000, chains=4, tune=4000, 
                                             idata_kwargs = {'log_likelihood': True}, 
                                             random_seed=rng)

az.plot_trace(trace_robust)
```

![The trace of the robust model](/docs/assets/images/statistics/robust_regression/trace_robust.webp)

The trace doesn't show relevant issues, so we can compute the posterior predictive.

```python
with model_robust:
    y_pred = pm.StudentT('y_pred', mu=alpha+x_plt*beta, sigma=1/tau, nu=nu)

with model_robust:
    ppc_robust = pm.sample_posterior_predictive(trace_robust, var_names=['y', 'y_pred'], random_seed=rng)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_plt, ppc_robust.posterior_predictive['y_pred'].median(dim=['draw', 'chain']))
ax.fill_between(x_plt, ppc_robust.posterior_predictive['y_pred'].quantile(q=0.025, dim=['draw', 'chain']),
                ppc_robust.posterior_predictive['y_pred'].quantile(q=0.975, dim=['draw', 'chain']), alpha=0.5, color='grey')
ax.scatter(df_turnout['turnout'].values, df_turnout['inequality'].values)
```

![The PPC of the robust model](/docs/assets/images/statistics/robust_regression/ppc_robust.webp)

This distribution does a better job in reproducing the data, but
it tells a very different story from the normal model.

While in fact in the above model an increase of the turnout
translated into a reduction of the average inequality,
with this robust model this conclusion does not appear so clearly.

Let us try and see what does the LOO can tell us.

```python
loo_normal = az.loo(trace_norm, model_norm)
loo_robust = az.loo(trace_robust, model_robust)

df_loo = az.compare({'Normal model': trace_norm, 'Robust model': trace_robust})

az.plot_compare(df_compare)
```

![The plot of the LOO cross-validation](/docs/assets/images/statistics/robust_regression/loo.webp)

The LOO is slightly better for the normal model,
they are however very similar. Let us try and understand why.

```python
df_compare
```

|              |   rank |   elpd_loo |   p_loo |   elpd_diff |   weight |      se |     dse | warning   | scale   |
|:-------------|-------:|-----------:|--------:|------------:|---------:|--------:|--------:|:----------|:--------|
| Normal model |      0 |   -30.8017 | 4.74402 |     0       | 0.879221 | 4.33551 | 0       | True      | log     |
| Robust model |      1 |   -32.452  | 6.97777 |     1.65029 | 0.120779 | 4.68686 | 2.10687 | False     | log     |

The difference is $1.65\,,$ and the difference due to the number
of the degrees of freedom is the difference of the $p_loo\,,$
which is approximately 2.2, so the entire preference is due
to the lower number of degrees of freedom of the normal distribution.

We can see, however, that the LOO estimate for the normal
model has a warning. This generally happens because the ELPD
estimate is not exact, and it's only reliable when 
removing one point does not affect too much log predictive density.

```python
loo_normal
```

<div class=code>
Computed from 16000 posterior samples and 18 observations log-likelihood matrix.
<br>
&nbsp; &nbsp; &nbsp; &nbsp;         Estimate       SE
<br>
elpd_loo   -30.80     4.34
<br>
p_loo        4.74        -
<br>

There has been a warning during the calculation. Please check the results.
<br>
- - - -
<br>

Pareto k diagnostic values:
<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Count   Pct.
<br>
(-Inf, 0.5]   (good)       16   88.9%
<br>
 (0.5, 0.7]   &nbsp; (ok)  &nbsp; 0    0.0%
<br>
 (0.7, 1]   &nbsp; (bad)  &nbsp;  2   11.1%
<br>
   (1, Inf)   (very bad)    0    0.0%
</div>

There are two points which strongly affect our parameters,
and one reasonable assumption is that one of those is the South Africa.

Let us try and see what does it happens once we remove it.

```
with pm.Model() as model_norm_red:
    alpha = pm.Normal('alpha', mu=0, sigma=20)
    beta = pm.Normal('beta', mu=0, sigma=20)
    tau = pm.HalfNormal('tau', sigma=5)
    y = pm.Normal('y', mu=alpha+df_turnout['turnout'].values[:17]*beta, observed=df_turnout['inequality'].values[:17], tau=tau)

with model_norm_red:
    trace_norm_red = pm.sample(draws=2000, chains=4, tune=2000,
                               idata_kwargs = {'log_likelihood': True},
                               random_seed=rng)

az.plot_trace(trace_norm_red)

```

![The trace for the new normal model](/docs/assets/images/statistics/robust_regression/trace_norm_red.webp)

```python
with model_norm_red:
    y_pred_red = pm.Normal('y_pred', mu=alpha+x_plt*beta, tau=tau)

with model_norm_red:
    ppc_norm_red = pm.sample_posterior_predictive(trace_norm_red, var_names=['y', 'y_pred'], random_seed=rng)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_plt, ppc_norm_red.posterior_predictive['y_pred'].mean(dim=['draw', 'chain']))
ax.fill_between(x_plt, ppc_norm_red.posterior_predictive['y_pred'].quantile(q=0.025, dim=['draw', 'chain']),
                ppc_norm_red.posterior_predictive['y_pred'].quantile(q=0.975, dim=['draw', 'chain']), alpha=0.5, color='grey')
ax.scatter(df_turnout['turnout'].values, df_turnout['inequality'].values)
```

![The PPC for the new model](/docs/assets/images/statistics/robust_regression/ppc_norm_red.webp)

This result looks much more to the robust estimate than to 
the full normal estimate.
While in the full normal model the parameter beta was
not compatible with 0, both for the robust and for the reduced
normal model it is.
This implies that those models contradict the full normal model,
which shows a negative association between the turnover
and the average income inequality.
Since the conclusion of the full normal model are heavily 
affected by the South Africa, before drawing
any conclusion one should carefully assess whether does this
makes sense. Is the South Africa really representative or
is it a special case? 

## Conclusions

We have discussed how to perform a robust linear regression,
and we have shown with an example that using it instead of a normal
linear regression makes our model more stable to the presence
of non-representative items.
