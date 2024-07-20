---
layout: post
title: "Difference in difference"
categories: /statistics/
subcategory: "Causal inference"
tags: /causal_intro/
date: "2024-06-30"
# image: "/docs/5ssets/images/perception/eye.jpg"
description: "Causal inference from 1850"
section: 3
---

Difference in differences is a very old technique,
and one of the first applications of
this method was done by John Snow, who's also
popular due to the cholera outbreak data visualization.

In his study, he used the **Difference in Difference**
(DiD) method to provide some evidence that,
during the London cholera epidemic of 1866,
the cholera was caused by drinking from a water
pump.
This method has been more recently used [by 
Card and Krueger in this work](https://davidcard.berkeley.edu/papers/njmin-aer.pdf)
to analyze the causal relationship between
minimum wage and employment.
In 1992, the New Jersey increased the minimum wage
from 4.25 dollars to 5.00 dollars.
They compared the employment in Pennsylvania
and New Jersey before and after the minimum wage increase
to assess if it caused a decrease in the New Jersey
occupation, as supply and demand theory would predict.

DiD assumes that, before the intervention $I$,
the untreated group and the treated one
both evolve linearly with the time $t$ with the
same slope,
while after the intervention the treated group
changes slope.
Assuming, that the intervention was applied at time
$t=0$ 

$$
\begin{align}
&
Y_{P}^0 = \alpha_{P} 
\\
&
Y_{P}^1 = \alpha_{P} +\beta
\\
&
Y_{NJ}^0 = \alpha_{NJ} 
\\
&
Y_{NJ}^1 = \alpha_{NJ} +\beta + \gamma
\end{align}
$$


In the above formulas, the intervention effect
is simply $\gamma\,.$

## The implementation

We downloaded the dataset from [this page](https://www.kaggle.com/code/harrywang/difference-in-differences-in-python/input).

```python
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import seaborn as sns
from matplotlib import pyplot as plt

df_employment = pd.read_csv('data/employment.csv')

sns.pairplot(data=df_employment, hue='state')
```

![The dataset pairplot](/docs/assets/images/statistics/difference_in_difference/pairplot.webp)

```python
df_before = df_employment[['state', 'total_emp_feb']]
df_after = df_employment[['state', 'total_emp_nov']]

# We will assign t=0 data before treatment and t=1 after the treatment
# Analogously g=0 will be the control group, g=1 will be the test group

df_before['t'] = 0
df_after['t'] = 1

df_before.rename(columns={'total_emp_feb': 'Y'}, inplace=True)
df_after.rename(columns={'total_emp_nov': 'Y'}, inplace=True)

df_before.rename(columns={'state': 'g'}, inplace=True)
df_after.rename(columns={'state': 'g'}, inplace=True)

df_reg = pd.concat([df_before, df_after])

## Let us build the interaction term

df_reg['gt'] = df_reg['g']*df_reg['t']

df_reg = df_reg[['g', 't', 'gt', 'Y']]

with pm.Model() as did_model:
    beta_0 = pm.Normal('beta_0', mu=0, sigma=1000)
    beta_g = pm.Normal('beta_g', mu=0, sigma=1000)
    beta_t = pm.Normal('beta_t', mu=0, sigma=1000)
    beta_gt = pm.Normal('beta_gt', mu=0, sigma=1000)
    sigma = pm.HalfCauchy('sigma', beta=50)
    nu = pm.HalfCauchy('nu', beta=200)
    mu = beta_0 + beta_g*df_reg['g']+ beta_t*df_reg['t']+ beta_gt*df_reg['gt']
    y = pm.StudentT('y', mu=mu, sigma=sigma, nu=nu,
                  observed=df_reg['Y'].values)
    trace_did = pm.sample(5000, tune=5000, chains=4)

az.plot_trace(trace_did)
```

![The model trace](/docs/assets/images/statistics/difference_in_difference/trace.webp)

The trace looks fine, let us now verify the posterior
predictive.

```python
with did_model:
    y00 = pm.StudentT('y00', mu=beta_0, sigma=sigma, nu=nu)
    y10 = pm.StudentT('y10', mu=beta_0+beta_g, sigma=sigma, nu=nu)
    y01 = pm.StudentT('y01', mu=beta_0+beta_t, sigma=sigma, nu=nu)
    y11 = pm.StudentT('y11', mu=beta_0+beta_g+beta_t+beta_gt, sigma=sigma, nu=nu)
    
    ppc_check = pm.sample_posterior_predictive(
        var_names=['y00','y01','y10','y11'], trace=trace_did)


fig, ax = plt.subplots(figsize=(8, 6), nrows=2, ncols=2)
for g in [0, 1]:
    for t in [0, 1]:
        ax[g][t].set_xlim([-20, 80])
        for i in range(50):
            sns.kdeplot(az.extract(data=ppc_check, var_names=[f"y{g}{t}"], group='posterior_predictive',num_samples=100), ax=ax[g][t],
                       color='lightgray')
        sns.kdeplot(az.extract(data=ppc_check, var_names=[f"y{g}{t}"], group='posterior_predictive'), ax=ax[g][t])
        sns.kdeplot(df_reg[(df_reg['g']==g)&(df_reg['t']==t)]['Y'], ax=ax[g][t])
        ax[g][t].set_title(f"g={g}, t={t}")

legend = ax[0][0].legend(frameon=False)
fig.tight_layout()
```

![The comparison between the predicted
and observed distributions of Y](/docs/assets/images/statistics/difference_in_difference/posterior_predictives.webp)

The posterior predictive distributions agree with the observed data. We extracted some random sub-sample to
provide an estimate of the uncertainties.

We can finally verify if there is any effect:

```python
az.plot_forest(trace_did, var_names=['beta_gt'])
```

![Our estimate for the minimum wage increase effect
](/docs/assets/images/statistics/difference_in_difference/effect_estimate.webp)

As you can see, the effect is compatible with 0, therefore there is no evidence
that by increasing the minimum salary there is an effect on the occupation.

Our model has a small issue: it allows for negative values of the occupation,
which doesn't make sense. This problem can be easily circumvented by using 
the [truncated PyMC class](https://www.pymc.io/projects/docs/en/v4.4.0/api/distributions/generated/pymc.Truncated.html).

I suggest you to try it and verify yourself if there is any effect.
Remember that in that case $\mu$ is no more the mean for $Y$,
so you can't use it to estimate the average effect.

## Conclusions

We have seen how to implement the DiD method with PyMC, and we used to
re-analyze the Krueger and Card article on the relation between the minimum
salary and the occupation.
