---
layout: post
title: "The autoregressive model"
categories: /statistics/
subcategory: "Time series"
tags: /time_series/
date: "2024-02-16"
# image: "/docs/5ssets/images/perception/eye.jpg"
description: "How to model dependence on the past"
section: 1
---

In the last post we introduced the time series, and in this post
we will look more in details to the autoregressive model,
namely

$$
y_k = \sum_{i=1}^r y_{k-i} + \varepsilon_k
$$

where $\varepsilon_k$ are iid.

We will use the airline passengers dataset.


```python
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
from scipy.signal import periodogram
import pymc.sampling_jax as pmj

df_pass = pd.read_csv("https://raw.githubusercontent.com/MakrandBhandari/Time-Series-Forecasting--Airline-Passengers-in-Python/main/international-airline-passengers.csv")
df_pass.rename(columns={"International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60": "Thous_pass"}, inplace=True)

df_pass = df_pass.iloc[:144]

fig = plt.figure()
ax = fig.add_subplot(111)
sns.lineplot(df_pass, x="Month", y="Thous_pass", ax=ax)
ax.set_xticks(df_pass['Month'].iloc[::24])
fig = plt.gcf()
fig.tight_layout()
```

![](/docs/assets/images/statistics/autoregressive/airline.webp)

A visual inspection of the data can be very useful, but sometimes it could not be 
enough to build a starting model. 
As an example, we can clearly see that there is a (probably linear) trend in our series,
that there is a strong annual periodicity, and that the amplitude of the periodic part increases
with the time.
There are however methods which may help us in this task.
The first is the [**periodogram**](https://en.wikipedia.org/wiki/Periodogram), which could help us in assessing the frequency of the
seasonal part.

```python
f, Pxx_spec = periodogram(df_pass['Thous_pass'], detrend='linear', scaling='spectrum')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.stem(f*12, np.sqrt(Pxx_spec)) # The dataset has monthly frequency, we now convert the frequency in Y^-1
ax.set_xlabel(r'frequency $[Y^{-1}]$')
ax.set_ylim([0, 35])
```

![The periodogram, where the frequency is expressed in inverse years](
/docs/assets/images/statistics/autoregressive/periodogram.webp)

The peak at 1 confirms us that there is a strong periodic component at one year, as well as
possibly some higher frequency components.

We now want to have an idea of the order of the autoregressive part, and
plotting the coefficients of the autocorrelation function can be helpful in this task.
While the periodogram automatically handles the linear component, here we must remove 
both the trend and the periodic components by hand. We can do this as follows:

```python
detrended = np.diff(df_pass['Thous_pass'].values)
deperiod = detrended[:-12] - detrended[12:]

fig = plt.figure()
ax = fig.add_subplot(211)
plot_acf(deperiod, ax=ax)
ax1 = fig.add_subplot(212)
plot_pacf(deperiod, ax=ax1)
fig.tight_layout()
```

![The ACF and PACF plot](/docs/assets/images/statistics/autoregressive/acorr.webp)

It looks like there is a small autoregressive component of order one.

We will now build a model with a trend, a periodic and a $$AR(1)$$ component.
We will start by only assuming a yearly component, and in order to reduce the
trend in the amplitude of the seasonal component, we will model the logarithm
of the number of passengers.
We will only use the first 120 points to fit the model, while the remaining 2 years will
be used to assess the performances of our model for future data.


```python
y_pass = (np.log(df_pass['Thous_pass'])).values
x_pass = np.arange(len(y_pass))

y_pass_fit = y_pass[:120]
x_pass_fit = np.arange(len(y_pass_fit))
n_pred = 144 - 120

with pm.Model() as pass_model:
    y0 = pm.Normal('y0', mu=0, sigma=10)
    alpha = pm.Normal('alpha', mu=0, sigma=2)
    beta = pm.Normal('beta', mu=0, sigma=0.5)
    gamma = pm.Normal('gamma', mu=0, sigma=0.5)
    eta = pm.Normal('eta', mu=0, sigma=5, shape=(2))
    sigma = pm.Exponential('sigma', lam=1)
    x_ar = pm.AR('x_ar', sigma=sigma, rho=eta, shape=len(y_pass_fit))
    muv = y0 + alpha*x_pass_fit + x_ar  +  beta*np.cos(2.0*np.pi*(x_pass_fit/12))+   gamma*np.sin(2.0*np.pi*(x_pass_fit/12))
    yhat_pass = pm.Normal('yhat_pass', mu=muv, sigma=sigma, observed=y_pass_fit)
    trace_pass = pmj.sample_numpyro_nuts()
```

![The trace plot](/docs/assets/images/statistics/autoregressive/trace.webp)

The trace looks OK, let us now verify if our model reproduces the fitted data

```python
with pass_model:
    ppc_pass = pm.sample_posterior_predictive(trace_pass)

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(df_pass['Month'], y_pass)
ax.plot(df_pass['Month'].iloc[:len(y_pass_fit)], ppc_pass.posterior_predictive['yhat_pass'].mean(dim=['chain', 'draw']))
ax.fill_between(df_pass['Month'].iloc[:len(y_pass_fit)], ppc_pass.posterior_predictive['yhat_pass'].quantile(dim=['chain', 'draw'], q=0.025), ppc_pass.posterior_predictive['yhat_pass'].quantile(dim=['chain', 'draw'], q=0.975),
               color='grey', alpha=0.5)
ax.set_xticks(df_pass['Month'].iloc[12::24])
fig.tight_layout()
```

![The PPC for the fit data](/docs/assets/images/statistics/autoregressive/ppc_fitted.webp)

The agreement looks good, except for some minor issues with high frequency modes which is
however not interesting for us, since we are not interested in such a high resolution.
We can now verify is our model is also able to reproduce the observed data for the last two years.

```python
x_pass_pred = x_pass[len(x_pass_fit):]

with pass_model:
    pass_model.add_coords({"z_1": range(len(y_pass_fit) - 1, len(y_pass), 1)})
    x_ar_pred = pm.AR(
            "x_ar_pred",
            init_dist=pm.DiracDelta.dist(x_ar[..., -1]),
            rho=eta,
            sigma=sigma,
            dims="z_1")
    periodic =  beta*np.cos(2.0*np.pi*(x_pass_pred/12))+gamma*np.sin(2.0*np.pi*(x_pass_pred/12))
    muv = y0 + alpha*x_pass_pred + x_ar_pred[1:]  +  periodic
    yhat_pass_pred = pm.Normal("yhat_pass_pred", mu=muv, sigma=sigma)
    
    ppc_ar_pred_y = pm.sample_posterior_predictive(trace_pass, var_names=['yhat_pass_pred'])

ypass_av = np.concatenate([
ppc_pass.posterior_predictive['yhat_pass'].mean(dim=['draw', 'chain']), ppc_ar_pred_y.posterior_predictive['yhat_pass_pred'].mean(dim=['draw', 'chain'])])
ypass_m = np.concatenate([
ppc_pass.posterior_predictive['yhat_pass'].quantile(q=0.025, dim=['draw', 'chain']), ppc_ar_pred_y.posterior_predictive['yhat_pass_pred'].quantile(q=0.025, dim=['draw', 'chain'])])
ypass_M = np.concatenate([
ppc_pass.posterior_predictive['yhat_pass'].quantile(q=0.975, dim=['draw', 'chain']), ppc_ar_pred_y.posterior_predictive['yhat_pass_pred'].quantile(q=0.975, dim=['draw', 'chain'])])

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(df_pass['Month'], y_pass)
ax.plot(df_pass['Month'], ypass_av)
ax.fill_between(df_pass['Month'], ypass_m, ypass_M,
               color='grey', alpha=0.5)
ax.set_xticks(df_pass['Month'].iloc[12::24])
ax.axvline(df_pass['Month'].iloc[len(y_pass_fit)], color='k', ls=':')
fig.tight_layout()
```

![](/docs/assets/images/statistics/autoregressive/ppc_all.webp)

We observe some discrepancy for the yearly minimum, but except for that the data
are always included in the credible interval.

We can now inspect the contribution of each component

```python
with pass_model:
    ppc_ar_pred_ar = pm.sample_posterior_predictive(trace_pass, var_names=['x_ar_pred'])

x_pass_all = np.concatenate([x_pass_fit, x_pass_pred])

y_trend = (trace_pass.posterior['y0'].mean(dim=['draw', 'chain']).values.reshape(-1)+trace_pass.posterior['alpha'].mean(dim=['draw', 'chain']).values.reshape(-1)*x_pass_all)
y_seas = (trace_pass.posterior['beta'].mean(dim=['draw', 'chain']).values.reshape(-1)*np.cos(2.0*np.pi*(x_pass_all/12))+trace_pass.posterior['gamma'].mean(dim=['draw', 'chain']).values.reshape(-1)*np.sin(2.0*np.pi*(x_pass_all/12)))
y_res = np.concatenate([trace_pass.posterior['x_ar'].mean(dim=['draw', 'chain']), ppc_ar_pred_ar.posterior_predictive['x_ar_pred'].mean(dim=['draw', 'chain'])[1:]])

fig = plt.figure()

ax = fig.add_subplot(311)
ax.plot(df_pass['Month'],y_trend)
ax.set_title('Trend')
ax.set_xticks(df_pass['Month'].iloc[12::24])
ax.axvline(x=df_pass['Month'].iloc[len(x_pass_fit)], color='k', ls=':')

ax1 = fig.add_subplot(312)
ax1.plot(df_pass['Month'],y_seas)
ax1.set_title('Seasonal')
ax1.set_xticks(df_pass['Month'].iloc[12::24])
ax1.axvline(x=df_pass['Month'].iloc[len(x_pass_fit)], color='k', ls=':')

ax2 = fig.add_subplot(313)
ax2.plot(df_pass['Month'],y_res)
ax2.set_title('Residual')
ax2.set_xticks(df_pass['Month'].iloc[12::24])
ax2.axvline(x=df_pass['Month'].iloc[len(x_pass_fit)], color='k', ls=':')

fig.tight_layout()
```

![](/docs/assets/images/statistics/autoregressive/components.webp)

As expected, the trend component is the most relevant one. We can also see that
the residual component has the same order of magnitude of the seasonal one.

## Conclusions

We have seen some tools which may help us choosing an appropriate model.
We have also seen how to implement a time series with an autoregressive component.
