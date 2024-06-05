---
layout: post
title: "Introduction to time series modelling"
categories: /statistics/
subcategory: "Time series"
tags: /time_series/
date: "2024-02-15"
# image: "/docs/5ssets/images/perception/eye.jpg"
description: "How to model time-dependent processes"
section: 0
---

Up to now we assumed that our samples were iid according to some
probability distribution. We will now modify this assumption
ad assume that each observation depends on the previous
observations.
We will assume that we are dealing with discrete time process,
and leave continuous time ones to a future post.

Let us take a look at some example of time series.

```python
import pandas as pd
import requests
import statsmodels as stats
import yfinance as yf
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
import statsmodels.api as sm
import pyreadr

df_co2 = pd.read_csv("https://raw.githubusercontent.com/gahjelle/data-analysis-with-python/master/data/co2-ppm-mauna-loa-19651980.csv")
df_co2.rename(columns={'CO2 (ppm) mauna loa, 1965-1980':"CO2_ppm"}, inplace=True)

df_stock = yf.download('RB=F', start='2021-01-01', end='2023-01-01', interval='1wk')

df_lynx = pd.read_csv('http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt', delim_whitespace=True, header=None, index_col=0)
df_lynx.index.name = 'Year'
df_lynx.columns = ['Hare', 'Lynx']

df_pass = pd.read_csv("https://raw.githubusercontent.com/MakrandBhandari/Time-Series-Forecasting--Airline-Passengers-in-Python/main/international-airline-passengers.csv")
df_pass.rename(columns={"International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60": "Thous_pass"}, inplace=True)
df_pass = df_pass.iloc[:144]

df_ssp = pd.read_csv('https://www.sidc.be/SILSO/INFO/snmtotcsv.php', sep=';', header=None)
df_sunspot = df_ssp.set_axis(['YEAR', 'month', 'ym', 'SUNACTIVITY', 'activity_sd', 'n_obs', 'is_definitive'], axis=1)
df_sunspot = df_sunspot[df_sunspot['YEAR']<2024]

with requests.get('https://github.com/cran/TSA/raw/master/data/larain.rda') as f:
    with open('./larain.rda', 'wb') as g:
        g.write(f.content)

df_rain = pyreadr.read_r('./larain.rda')

df_larain = df_rain['larain']

df_larain['year'] = range(1878,1993)

fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
sns.lineplot(df_co2, x="Month", y="CO2_ppm", ax=ax[0][0])
ax[0][0].set_xticks(df_co2['Month'].iloc[::24])
sns.lineplot(df_stock, x="Date", y="Close", ax=ax[0][1])
sns.lineplot(df_lynx, x="Year", y="Lynx", ax=ax[1][0])
sns.lineplot(df_pass, x="Month", y="Thous_pass", ax=ax[1][1])
ax[1][1].set_xticks(df_pass['Month'].iloc[::24])
sns.lineplot(df_sunspot, x="YEAR", y="SUNACTIVITY", ax=ax[2][0])
sns.lineplot(df_larain, x="year", y="larain", ax=ax[2][1])
fig.tight_layout()
```

![Some time series example
](/docs/assets/images/statistics/time_series_intro/ts_examples.webp)

Each time series has different characteristics, and each of them encodes different
features.

First of all, we can classify the time series depending on its **trend**, that is
the presence of a monotone growth or decrease of the mean.
This component is clearly visible in the airline dataset (4), in the $CO_2$ one (1) and in the stock
(number 2, where we can see a growth followed by a decrease).

Another kind of common component is the **seasonal** (or periodic) one,
which is present if there's some recurrent pattern.
This component is evident in the $CO_2$ and in the airline dataset, but also possibly in the lynx one
as well as in the sunspot data.

A common approach is to decompose the time series into a sum of trend, seasonality and residual random component,
and not all of them are always needed.
As an example, the $CO_2$ time series can be well reproduced without its random component,
while the inclusion of this part will the core topic of the future posts in this section.

## The $$CO_2$$ series

We will only use a subset of our sample to fit the model, and use the remaining
point to check the performance of our model for this subset of points.

```python
cut_df = 150

y = df_co2['CO2_ppm'].iloc[:cut_df]/1000
X = np.arange(len(y))

y_all = df_co2['CO2_ppm']/1000
X_all = np.arange(len(y_all))

with pm.Model() as co2_model:
    alpha = pm.Normal('alpha', mu=0, sigma=3)
    beta = pm.Normal('beta', mu=0, sigma=3)
    sigma = pm.Exponential('sigma', lam=10)
    gamma = pm.Normal('gamma', mu=0, sigma=3)
    Xp = np.sin(2.0*np.pi*X/12)  # The regressor for the periodic component
    mu0 = alpha+beta*X + gamma*Xp
    # X is the regressor for the trend component
    y_co2 = pm.Normal('y_co2', mu=mu0, sigma=sigma, observed=y)

with co2_model:
    trace_co2 = pmj.sample_numpyro_nuts(tune=20000, draws=20000, target_accept=0.9)

az.plot_trace(trace_co2)
```

![The trace of the CO2 model
](/docs/assets/images/statistics/time_series_intro/trace_co2.webp)

Let us now extent the model above the fitted data

```python
with co2_model:
    Xp_all = np.sin(2.0*np.pi*X_all/12)
    mu0_all = alpha+beta*X_all + gamma*Xp_all
    y_co2_pred = pm.Normal('y_co2_pred', mu=mu0_all, sigma=sigma)
    ppc_co2 = pm.sample_posterior_predictive(trace_co2, var_names=['y_co2_pred'])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.fill_between(
    df_co2['Month'],
ppc_co2.posterior_predictive['y_co2_pred'].quantile(q=0.025, dim=['draw', 'chain']),
ppc_co2.posterior_predictive['y_co2_pred'].quantile(q=0.975, dim=['draw', 'chain']),
    color='lightgray', alpha=0.8
)
ax.plot(
    df_co2['Month'],
ppc_co2.posterior_predictive['y_co2_pred'].mean(dim=['draw', 'chain']))

ax.plot(df_co2['Month'], y_all, ls=':')
ax.axvline(x=df_co2['Month'].iloc[cut_df], color='k', ls=':')
ax.set_xticks(df_co2['Month'].iloc[::24])
```

![The posterior predictive of the CO2 model](/docs/assets/images/statistics/time_series_intro/co2.webp)

We are using only the data below the black dotted line to fit the model.
As we can see, our model reproduces with quite a good accuracy also the data above
this line for many months.
When you have enough data it can be really useful to perform this check, in order to ensure
that your model is able to reproduce the observed data for the future,
at least with the current knowledge.

## Conclusions

We have seen a possible decomposition of a time series, and we saw that a possible
way to deal with the seasonal and trend components is to treat them as we would
do in a normal inference problem.
We saw how to split our data in order to check the performance of our model for unobserved data.
