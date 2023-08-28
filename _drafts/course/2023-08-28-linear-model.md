---
layout: page
title: "The linear model"
categories: course/intro/
tags: /linear_model/
---

In this post we will start looking at some regression problem,
 which are models where we want to relate the behavior of the outcome
 variable $y$ to some other variable $x$ which we do not want to model.
In particular, we will look the most fundamental regression model, the linear model.
This model is so widespread that entire statistical textbooks
and academic courses has been devoted to it, and it is crucial to fully
understand both how to assess and give a correct interpretation of the uncertainties
in this model and how to report these uncertainties no non-statisticians.
Of course we will just give few examples, without any claim of completeness.
For a full immersion in this model from a frequentist
perspective I reccomend the Weisberg textbook
"Applied linear regression", freely available
[here](https://www.stat.purdue.edu/~qfsong/teaching/525/book/Weisberg-Applied-Linear-Regression-Wiley.pdf).

## Normal linear regression

Consider the following dataset, describing the lung capacity of a set of patients. The most relevant covariate here is the age, but there are also other possible relevant quantities, and we will consider them later.

```python
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pymc.sampling_jax as pmjax


plt.style.use("seaborn-v0_8-darkgrid")

df_lungs = pd.read_csv('https://raw.githubusercontent.com/tkseneee/Dataset/master/LungCapdata.csv')

df_lungs.head()
```

|    |   Age |   Height |   Gender |   Smoke |   FEV |
|---:|------:|---------:|---------:|--------:|------:|
|  0 |     9 |     57   |        0 |       0 | 1.708 |
|  1 |     8 |     67.5 |        0 |       0 | 1.724 |
|  2 |     7 |     54.5 |        0 |       0 | 1.72  |
|  3 |     9 |     53   |        1 |       0 | 1.558 |
|  4 |     9 |     57   |        1 |       0 | 1.895 |


Here FEV means Forced Expiratory Volume, and measures how much air a person can exhale during a forced breath.

```python
sns.pairplot(df_lungs)
```

![Lung pairplot](/docs/assets/images/linear_model/lung_pairplot.jpg)

As we could imagine, there is a linear correlation between the age and the FEV.
While the age seems almost normally distributed, the FEV is not,
and as well the FEV variance grows with the age.
The distribution of the FEV seems definitely different between
the smoke=0 and the smoke=1 patients, so we should also take this into account.
But let us first start with the simplest linear model,
where we assume a linear relation between the FEV and the age.

```python
x_0 = df_lungs['Age'].values

with pm.Model() as linear_model_0:
    theta_0 = pm.Normal('theta_0', mu=0, sigma=1)
    theta_1 = pm.Normal('theta_1', mu=0, sigma=1)
    sigma = pm.HalfCauchy('sigma', beta=1)
    mu = theta_0 + theta_1*x_0
    y = pm.Normal('y', mu=mu, sigma=sigma,
                  observed=df_lungs['FEV'].values)

    trace_lm0 = pm.sample(2000, tune=500, chains=4,
    return_inferencedata=True, random_seed=np.random.default_rng(42))
```

Let us now check if we can spot any problem in the sampling procedure:

```python
az.plot_trace(trace_lm0)
```

![Lung trace](/docs/assets/images/linear_model/lung_trace.jpg)

```python
az.plot_autocorr(trace_lm0)
```

![Lung corrplot](/docs/assets/images/linear_model/lung_corrplot.jpg)

```python
az.plot_rang(trace_lm0)
```

![Lung rank](/docs/assets/images/linear_model/lung_rank.jpg)

```python
az.summary(trace_lm0)
```

|         |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:--------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| theta_0 |  0.428 | 0.078 |    0.285 |     0.574 |       0.001 |     0.001 |       2815 |       3553 |       1 |
| theta_1 |  0.222 | 0.007 |    0.208 |     0.236 |       0     |     0     |       2808 |       3506 |       1 |
| sigma   |  0.568 | 0.016 |    0.54  |     0.6   |       0     |     0     |       4048 |       3554 |       1 |

So far so good:
- The traces look fine
- The correlation goes to 0 after few iterations for all the variables
- The rank plots look almost uniform
- r_hat is 1 and the ESS are large enough.

Of course, we don't expect out model to be able to exactly reproduce
all the relevant features of the FEV plot, but let us check how far away is it.
In order to do this, we will take 20 random samples and compare them with the true sample:

```python
pp_lm0 = pm.sample_posterior_predictive(model=linear_model_0, trace=trace_lm0, random_seed=np.random.default_rng(42))

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(20):
    s = np.random.randint(2000)
    t = np.random.randint(4)
    ax.scatter(x_0, pp_lm0.posterior_predictive.y.values[t][s], marker='+', color='navy')
ax.scatter(x_0, df_lungs['FEV'].values, color='green')
ax.set_xlabel('AGE')
ax.set_ylabel('FAV  ', rotation=0)
fig.tight_layout()
```

![Lung PPC](/docs/assets/images/linear_model/lung_ppc.jpg)

Our model overestimates the uncertainties for lower age values, up to 10 years or so, but apparently it catches all the other relevant features of the sample.
In the notebook on causal inference we will see how to deal with data with
non-constant variance [^1], as it happens in the previous plot.

[^1]: Data with constant variance are called _homoskedastik_ while data with varying variance are called _heteroskedastik_.
