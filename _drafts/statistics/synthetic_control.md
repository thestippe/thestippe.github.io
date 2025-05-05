---
layout: post
title: "Synthetic control"
categories: /statistics/
subcategory: "Causal inference"
tags: /synthetic_control/
date: "2024-07-14"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Building a doppelganger from the control group"
section: 5
published: false
---

The **synthetic control** method recently became a very popular method
among economists (although I honestly can't see the same enthusiasm in
the statistics community).
This method has been widely (and a little bit wildly) used to assess
the effects on a quantity $$Y^{\bar{s}}_t$$ of the introduction of a new policy into a country $s$
(or other geographical region) at a time $t=t_1$.
Assuming that you have the same quantity for a set of similar countries $s_i$
as well as for the target country $\bar{s}\,,$
you assume that the time behavior of $$Y_{\bar{s}} = (Y_{t_0}^{\bar{s}}, \dots, Y^{\bar{s}}_{t_1})$$ before the intervention is given by a weighted
average of $Y^{s_i}\,.$

You moreover assume, as control, the same weighted average
$$\bar{Y}^{\bar{s}}$$ after the intervention.

A very detailed discussion of this method can be found on [Juan Camilo Orduz' page](https://juanitorduz.github.io/synthetic_control_pymc/).
We will use the same model, but we will apply it to a different dataset.

While in fact he uses PyMC to reproduce [this example](https://matheusfacure.github.io/python-causality-handbook/landing-page.html),
we will use it to perform a simplified re-analysis of [this article](https://link.springer.com/article/10.1007/s10584-021-03111-2), where the authors analyze the impact of the introduction
of a policy for the reduction of the $CO_2$ emissions in the UK.
The dataset used in this work can be found [on Zenodo](https://zenodo.org/records/4566804).

The authors of the original work, in fact, performed a careful analysis
of the control set, while we will limit ourselves to the set of countries
who were in the OECD organization in 2001 and who had not adopted any 
$CO_2$ reduction policy before that year.
We will assume

$$
Y^{\bar{s}} \sim \mathcal{N}(\mu, \sigma)
$$

In order to ensure that the behavior before the intervention is carefully
reproduced, we assume a small variance

$$
\sigma \sim \mathcal{Exp}(100)
$$

As anticipated, $\mu$ is given by

$$
\mu = \sum_{i=1}^n \omega_{i} Y^i
$$

We assume that the weights sum up to one, so we assume

$$
\omega \sim \mathcal{Dir}(1/n,\dots,1/n)
$$

```python
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
from matplotlib import pyplot as plt

rng = np.random.default_rng(123454321)

df_dt = pd.read_csv('./data/climate_policies.csv', sep=';')
df_carb = pd.read_csv('./data/nation.1751_2014.csv')

df_red = df_carb[df_carb['Year']>=1990][['Nation', 'Year', 'Per capita CO2 emissions (metric tons of carbon)']]

df_co2 = df_red.pivot(index='Year', columns='Nation', values='Per capita CO2 emissions (metric tons of carbon)')

# Taken from the repo

oecd = ["AUSTRALIA","AUSTRIA","BELGIUM","CANADA","CZECH REPUBLIC",
        "DENMARK","FINLAND","FRANCE (INCLUDING MONACO)","GERMANY",
        "GREECE","HUNGARY","ICELAND","IRELAND","ITALY (INCLUDING SAN MARINO)",
        "JAPAN","LUXEMBOURG","MEXICO","NETHERLANDS","NEW ZEALAND","NORWAY",
        "POLAND","PORTUGAL","SLOVAKIA","REPUBLIC OF KOREA","SPAIN","SWEDEN",
        "SWITZERLAND","TURKEY","UNITED KINGDOM","UNITED STATES OF AMERICA"]

to_exclude = ['DENMARK', 'ESTONIA', 'FINLAND', 'NETHERLANDS', 'NORWAY',
       'SLOVENIA', 'SWEDEN', "UNITED KINGDOM"]

donors = list(set(oecd)-set(to_exclude))

df_in = df_co2[donors].dropna(axis=1)

with pm.Model(coords={'countries': df_in.columns, 'years':np.arange(1990, 2001)}) as sc_model:
    lam = pm.Gamma('lam', 1/len(df_in.columns)**2, 1/len(df_in.columns))
    w = pm.Dirichlet('w', a=np.ones(len(df_in.columns))*lam, shape=(len(df_in.columns)),
                     dims=['countries'])
    sigma = pm.Exponential('sigma', lam=100)
    mu = 0
    for k, col in enumerate(df_in.columns):
        mu += w[k]*df_in[col].loc[1990:2001].astype(float)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=df_co2['UNITED KINGDOM'].loc[1990:2001])

with sc_model:
    trace = pm.sample(chains=4, draws=5000, tune=5000,
                     nuts_sampler='numpyro', target_accept=0.98,
                     random_seed=rng)

az.plot_trace(trace)
fig = plt.gcf()
fig.tight_layout()
```
![The trace plot](/docs/assets/images/statistics/synthetic_control/trace.webp)

We ran quite a large number of draws as the number of parameters is quite large
and rather unconstrained. However, the trace looks fine.
An important thing that one should always verify when using
a synthetic control, is that the weights must be sparse (only few should
dominate, while the remaining should be close to 0).

```python
az.plot_forest(trace, var_names=['w'])
fig = plt.gcf()
fig.tight_layout()
```

![The forest plot of the weights](/docs/assets/images/statistics/synthetic_control/weights.webp)

The requirement seems fulfilled, as only few dominate the entire fit.
We can now compute the posterior predictive before and after the intervention.

```python
with sc_model:
    mu1 = 0
    for k, col in enumerate(df_in.columns):
        mu1 += w[k]*df_in[col].loc[2002:].astype(float)
    y1 = pm.Normal('y1', mu=mu1, sigma=sigma)
    ppc = pm.sample_posterior_predictive(trace, var_names=['y', 'y1'])

yv = np.concatenate([ppc.posterior_predictive['y'].values.reshape((20000, -1)),
                     ppc.posterior_predictive['y1'].values.reshape((20000, -1))],
                     axis=1)

fig, ax = plt.subplots()
ax.spines[['right', 'top']].set_visible(False)
uk = ax.plot(df_co2.index, df_co2['UNITED KINGDOM'].values.astype(float), label='UK')
ax.fill_between(df_co2.index, np.quantile(yv, axis=0, q=0.025), np.quantile(yv, axis=0, q=0.975), color='grey', alpha=0.5)
synth = ax.plot(df_co2.index, np.mean(yv, axis=0), label='Synthetic UK')
ax.axvline(x=2001, color='k', ls=':')
ax.set_xlabel("Year")
ax.set_title(r"Per capita $CO_2$ $m^3/Year$")
ax.annotate('Synthetic', xy=(df_co2.index[-1], np.mean(yv, axis=0)[-1]), color=synth[0].get_color() )
ax.annotate('UK', xy=(df_co2.index[-1], df_co2['UNITED KINGDOM'].values.astype(float)[-1]), color=uk[0].get_color() )
```

![The comparison between the true and the synthetic UK](/docs/assets/images/statistics/synthetic_control/posterior_predictive.webp)

As we can see, the behavior is very similar up to 2001, while after this date
the synthetic UK $CO_2$ consumption is larger than one of the true $UK\,.$
You can verify yourself that, by only fitting up to 2000, the result doesn't
change, and the lines still diverge starting from 2002.
This is another important check that you should always perform when using the
synthetic control method.

## Conclusion

We have seen how to implement the synthetic control method, together with
some of the most important checks that you should always do in order to
exclude major problems in your model.
We also re-analyzed [this article](https://link.springer.com/article/10.1007/s10584-021-03111-2), obtaining the same conclusions.


## Suggested readings

- <cite>Alice Lépissier & Matto Mildenberger, 2021. <A HREF="https://ideas.repec.org/a/spr/climat/v166y2021i3d10.1007_s10584-021-03111-2.html">Unilateral climate policies can substantially reduce national carbon pollution</A>, <A HREF="https://ideas.repec.org/s/spr/climat.html">Climatic Change</A>, Springer, vol. 166(3), pages 1-21, June.</cite>

```python
%load_ext watermark
```

```python
%watermark -n -u -v -iv -w -p xarray,numpyro,jax,jaxlib
```

<div class="code">
Last updated: Tue Aug 20 2024
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
numpy     : 1.26.4
<br>
pandas    : 2.2.2
<br>
pymc      : 5.15.0
<br>
matplotlib: 3.9.0
<br>
arviz     : 0.18.0
<br>

<br>
Watermark: 2.4.3
<br>
</div>