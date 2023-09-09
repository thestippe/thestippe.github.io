---
layout: post
title: "The linear model"
categories: course/intro/
tags: /linear-model/
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

Consider the [following dataset](https://search.r-project.org/CRAN/refmans/mplot/html/fev.html),
describing the lung capacity of a set of 654 young patients with age ranging from 3 to 19, 
recorded in a series of measures performed in the 1970s.
This dataset was used in [this](https://pubmed.ncbi.nlm.nih.gov/463860/) article to investigate the effect of having a smoking
parent on the respiratory capacity of the children.
The most relevant covariate here is the age, but there are also other possible relevant quantities, and we will consider them later.

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


Here FEV means Forced Expiratory Volume, and roughly measures how many liters of air a person can exhale in the first second of forced breath.

```python
sns.pairplot(df_lungs)
```

![Lung pairplot](/docs/assets/images/linear_model/lung_pairplot.jpg)

As we could imagine, the FEV depends on the age.
While the age seems almost normally distributed, the FEV is not,
and as well the FEV variance grows with the age.
The distribution of the FEV seems definitely different between
the smoke=0 and the smoke=1 patients, so we should also take this into account.

Let us give a closer look to the data we are going to fit:


```python
x_0 = df_lungs['Age'].values

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_0, df_lungs['FEV'].values, color='green')
ax.set_xlabel('AGE [Y]')
ax.set_ylabel('FAV [L]')
fig.tight_layout()
```

![Lung data](/docs/assets/images/linear_model/lung_data.jpg)

The relation between the age and the FEV looks linear, 
so let us try and fit the data with the linear model,
where we assume a linear relation between the FEV and the age.

```python
with pm.Model() as linear_model_0:
    theta_0 = pm.Normal('theta_0', mu=0, sigma=1)
    theta_1 = pm.Normal('theta_1', mu=0, sigma=1)
    sigma = pm.HalfCauchy('sigma', beta=1)
    mu = theta_0 + theta_1*x_0
    y = pm.Normal('y', mu=mu, sigma=sigma,
                  observed=df_lungs['FEV'].values)
```

Since we do not want to use our priors to constrain too much our model, we used uninformative priors for all the parameters.
Let us verify that our prior guess includes the data

```python
with linear_model_0:
    lm_prior_pred = pm.sample_prior_predictive()

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(20):
    t = np.random.randint(t, high=500)
    ax.scatter(x_0, lm_prior_pred.prior_predictive.y.values[0,t,:], marker='+', color='navy')
ax.scatter(x_0, df_lungs['FEV'].values, color='green')
ax.set_xlabel('AGE [Y]')
ax.set_ylabel('FAV  [L]')
fig.tight_layout()
```

![Lung prior predictive](/docs/assets/images/linear_model/lung_prior_pred.jpg)

There is not any observed point which is not covered by the prior predictive, so we can be confident that
our prior are generous enough to reproduce the observed data.
Now that we ensured that our prior guess looks OK, we can proceed with the next step
and perform the sampling of the posterior distribution.


```python
with linear_model_0:
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
ax.set_xlabel('AGE [Y]')
ax.set_ylabel('FAV [L]')
fig.tight_layout()
```

![Lung PPC](/docs/assets/images/linear_model/lung_ppc.jpg)

Our model overestimates the uncertainties for lower age values, up to 10 years or so, but apparently it catches all the other relevant features of the sample.
When we will discuss about causal inference we will see how to deal with data with
non-constant variance [^1], as it happens in the previous plot.

### Interpretation

We have three parameters in the model:
- $\theta_0\,,$ which represents how our model would predict the average FEV of a newborn child
- $\theta_1\,,$ which represents the average slope of the FEV. Alternatively, we can think about this parameter as how does the average FEV changes when we increment the age by one.
- $\sigma\,,$ which is the variance of the FEV. Notice that this is assumed to be independent on the age.


In our dataset there are no points for 0 years old children, so you could ask yourself if you are allowed to claim that $\theta_0$ is actually an estimate
for the FEV of a newborn child. There are many risks in doing so: does the current knowledge regarding the lung growth in babies allow you to assume that
the FEV can be extrapolated in a linear way? You should only extrapolate if your model is robust enough.
You can easily convince yourself about this by trying to extrapolate in the other direction. Since, in the linear model, the intercept $\theta_0$ becomes always less relevant
as $x$ grows, we may roughly say that the average of a 20 years old person is twice of the FEV of a 10 years old person, and this makes sense,
as our measures say that the average FEV for a 10 years old children is around 2.5, while the one of a 20 years old person is roughly 5.
But if we iterate this reasoning we would say that the average FEV of a 40 years old person is twice of the FEV of a 20 years old person,
and the FEV of an 80 years old person is four times the FEV of a 20 years old person. Of course, this simply sounds crazy, since 
we expect that elderly people will have a lower FEV than younger adults.


## Robust linear regression

In some case your data may be not good enough to provide you reliable estimates with normal linear regression,
and this is the case of the conclusions drawn from
[this](https://www.cambridge.org/core/journals/american-political-science-review/article/abs/political-institutions-and-voter-turnout-in-the-industrial-democracies/D6725BBF93F2F90F03A69B0794728BF7) article,
where the author concludes that there is a significant correlation between the voter turnout in a country and its average income inequality.
This example is a classical example of misleading result of a regression, where the author does not provide a plot of the data,
taken from [Healy, "Data visualization, a practical introduction"](https://www.google.it/books/edition/Data_Visualization/3XOYDwAAQBAJ?hl=it&gbpv=1&dq=Data+visualization,+a+practical+introduction&printsec=frontcover).

```python
df_turnout = pd.read_csv('data/inequality.csv')
df_turnout.head()
```

|    |   turnout |   inequality |
|---:|----------:|-------------:|
|  0 |  0.85822  |      1.95745 |
|  1 |  0.837104 |      1.95745 |
|  2 |  0.822021 |      2.41135 |
|  3 |  0.87632  |      2.76596 |
|  4 |  0.901961 |      2.95035 |

In this dataset we have on the percentage turnout against the average income inequality.
How can we decide, from a Bayesian perspective, if the conclusion hold? As already pointed
out, we cannot rely on statistical significance.
For this kind of problem we can use the ROPE, which is the Region Of Practical Equivalence:
before looking at the data, we should decide a region such that, if the HDI of a certain parameter
is included inside the region, we will conclude that the parameter is negligible. 
In our model we will be interested with the slope of the fitted model.
We decide that, if the HDI is included between $[-5, 5]$, then the slope is compatible with
0, so a change in the turnout does not lead to a large enough change into average the inequality income.


```python
az.pairplot(df_turnout)
```

![Turnout pairplot](/docs/assets/images/linear_model/inequality_pairplot.jpg)

By simply plotting the data we can clearly see that there is one point, the South Africa, which is far away from the other, and this may have a huge impact on the fit.
Let us see this, and how one may avoid this kind of error.

```python
with pm.Model() as model_norm:
    alpha = pm.Normal('alpha', mu=0, sigma=20)
    beta = pm.Normal('beta', mu=0, sigma=20)
    tau = pm.HalfNormal('tau', sigma=5)
    y = pm.Normal('y', mu=alpha+df_turnout['turnout'].values*beta, observed=df_turnout['inequality'].values, tau=tau)
    trace_norm = pm.sample(draws=2000, chains=4, tune=2000, idata_kwargs = {'log_likelihood': True}, random_seed=np.random.default_rng(42))
az.plot_trace(trace_norm)
```

![Traceplot normal](/docs/assets/images/linear_model/trace_norm.jpg)

```python
x_plt = np.arange(0, 1, 0.001)
with model_norm:
    y_pred = pm.Normal('y_pred', mu=alpha+x_plt*beta, tau=tau)
    ppc_norm = pm.sample_posterior_predictive(trace_norm, var_names=['y', 'y_pred'], random_seed=np.random.default_rng(42))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_plt, ppc_norm.posterior_predictive['y_pred'].mean(dim=['draw', 'chain']))
ax.fill_between(x_plt, ppc_norm.posterior_predictive['y_pred'].quantile(q=0.025, dim=['draw', 'chain']), ppc_norm.posterior_predictive['y_pred'].quantile(q=0.975, dim=['draw', 'chain']), alpha=0.5, color='grey')
ax.scatter(df_turnout['turnout'].values, df_turnout['inequality'].values)
```

![PPC normal](/docs/assets/images/linear_model/inequality_norm_ppc.jpg)

The error bands include all the points and it looks like it correctly reproduces the data. 
Does this model support the conclusions of the cited article?

```python
az.plot_forest(trace_norm, var_names=['beta'], rope=[-5, 5])
```

![PPC normal](/docs/assets/images/linear_model/inequality_forest_normal.jpg)

Yes, this model drives us to the same conclusions of the above cited article.
Let us now try with a more robust model, using both for the prior and
for the likelihood distribution with heavier tails than the normal distribution.


```python
with pm.Model() as model_robust:
    alpha = pm.Cauchy('alpha', alpha=0, beta=20)
    beta = pm.Cauchy('beta', alpha=0, beta=20)
    tau = pm.HalfCauchy('tau', beta=5)
    nu = pm.Exponential('nu', lam=1)
    y = pm.StudentT('y', mu=alpha+df_turnout['turnout'].values*beta, observed=df_turnout['inequality'].values, sigma=1/tau, nu=nu)
    trace_robust = pmjax.sample_numpyro_nuts(draws=2000, chains=4, tune=2000, idata_kwargs = {'log_likelihood': True},
    target_accept=0.98, random_seed=np.random.default_rng(42))

az.plot_trace(trace_robust)
```


![Traceplot robust](/docs/assets/images/linear_model/inequality_trace_robust.jpg)

```python
with model_robust:
    y_pred = pm.StudentT('y_pred', mu=alpha+x_plt*beta, sigma=1/tau, nu=nu)
    ppc_robust = pm.sample_posterior_predictive(trace_robust, var_names=['y', 'y_pred'], random_seed=np.random.default_rng(42))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_plt, ppc_robust.posterior_predictive['y_pred'].median(dim=['draw', 'chain']))
ax.fill_between(x_plt, ppc_robust.posterior_predictive['y_pred'].quantile(q=0.025, dim=['draw', 'chain']), ppc_robust.posterior_predictive['y_pred'].quantile(q=0.975, dim=['draw', 'chain']), alpha=0.5, color='grey')
ax.scatter(df_turnout['turnout'].values, df_turnout['inequality'].values)
```

![PPC robust](/docs/assets/images/linear_model/inequality_ppc_robust.jpg)

Also in this case the model correctly reproduces the data, but the points now are located far away from the limits of the error bands
and the slope significantly decreased.


```python
az.plot_forest(trace_robust, var_names=['beta'], rope=[-5, 5])
```

![Forest robust](/docs/assets/images/linear_model/inequality_forest_robust.jpg)

In this case the ROPE is only partially compatible with the HDI, so we cannot draw any conclusion.

Let us now check which model is the best one. We will do this by using the "LOO - Leave One Out" metric (see the model evaluation post).
The LOO function returns many results, included the Pareto shape values for each observation.
The Pareto index ranges from $-\infty$ to $\infty$ and the bigger it is, the less likely the observation is.
One usually takes all the observations above $0.7$ as bad observations, and the ones above $1$ are considered as extremely bad,
while the ones below $0.5$ are good.

```python
loo_normal = az.loo(trace_norm, model_norm)
loo_robust = az.loo(trace_robust, model_robust)

```

```python
loo_normal
```

| Pareto k   | Meaning | Count | Pct |
|------------|---------|-------|-----|
| (-Inf, 0.5]| Good    | 16    | 88.9|
|  (0.5, 0.7]| Ok      | 1     | 5.6 |
|  (0.7, 1.0]| Bad     | 0     | 0   |
|  (1, Inf)  | Very bad| 1     | 5.6 |

```python
loo_robust
```

| Pareto k   | Meaning | Count | Pct |
|------------|---------|-------|-----|
| (-Inf, 0.5]| Good    | 17    | 94.4|
|  (0.5, 0.7]| Ok      | 1     | 5.6 |
|  (0.7, 1.0]| Bad     | 0     | 0   |
|  (1, Inf)  | Very bad| 0     | 0   |


If we recall what we wrote about the LOO diagnostic and we observe the summary of the normal model,
we have that there is one point in the dataset such that,
when removed from the fit, becomes highly unlikely.
On the opposite, as expected, this does not happens for the robust model, as the Student-t distribution
can easily accommodate more extreme values without being affected as much as the normal model.

Let us see what does the model comparison tell us.

```python
az.compare({'Normal model': trace_norm, 'Robust model': trace_robust})
```

|              |   rank |   elpd_loo |   p_loo |   elpd_diff |    weight |      se |     dse | warning   | scale   |
|:-------------|-------:|-----------:|--------:|------------:|----------:|--------:|--------:|:----------|:--------|
| Normal model |      0 |   -30.8054 | 4.73068 |     0       | 0.920135  | 4.28969 | 0       | True      | log     |
| Robust model |      1 |   -32.7588 | 7.15545 |     1.95346 | 0.0798655 | 4.63041 | 2.12229 | False     | log     |

Both our models are able to correctly reproduce the data, but there is a strong penalty for the robust model for the extra parameter.
Moreover, the comparison shows a warning, which may be due to the presence of an outlier.
At this point a careful researcher would try and remove the problematic observation and see what does it happen to the estimates of each model.

```python
with pm.Model() as model_norm_red:
    alpha = pm.Normal('alpha', mu=0, sigma=20)
    beta = pm.Normal('beta', mu=0, sigma=20)
    tau = pm.HalfNormal('tau', sigma=5)
    y = pm.Normal('y', mu=alpha+df_turnout['turnout'].values[:17]*beta, observed=df_turnout['inequality'].values[:17], tau=tau)
    trace_norm_red = pm.sample(draws=2000, chains=4, tune=2000, idata_kwargs = {'log_likelihood': True}, random_seed=np.random.default_rng(42))

az.plot_trace(trace_norm_red)
```


![Trace normal reduced model](/docs/assets/images/linear_model/inequality_trace_normal_red.jpg)

```python
with model_norm_red:
    y_pred_red = pm.Normal('y_pred', mu=alpha+x_plt*beta, tau=tau)
    ppc_norm_red = pm.sample_posterior_predictive(trace_norm_red, var_names=['y', 'y_pred'], random_seed=np.random.default_rng(42))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_plt, ppc_norm_red.posterior_predictive['y_pred'].mean(dim=['draw', 'chain']))
ax.fill_between(x_plt, ppc_norm_red.posterior_predictive['y_pred'].quantile(q=0.025, dim=['draw', 'chain']), ppc_norm_red.posterior_predictive['y_pred'].quantile(q=0.975, dim=['draw', 'chain']), alpha=0.5, color='grey')
ax.scatter(df_turnout['turnout'].values, df_turnout['inequality'].values)
```

![PPC normal reduced model](/docs/assets/images/linear_model/inequality_ppc_normal_red.jpg)

By explicitly removing the South Africa point, the fit changes in a dramatic way, as beta becomes compatible with zero and the South Africa is no more 
included into the 95% error bands.

```python
az.plot_forest(trace_norm_red, var_names=['beta'], rope=[-5, 5])
```

![Forest normal red](/docs/assets/images/linear_model/inequality_forest_normal_red.jpg)

This model explicitly contradicts the first model, and tells us that there when you excludes the South Africa from the dataset
you won's see any association between turnout and average income inequality.
By seeing this result, one should investigate why the South Africa has a behavior which is so different from the one of the other
countries, and only after a sensible answer to this question one should decide if he wants to include this
point inside the dataset.
