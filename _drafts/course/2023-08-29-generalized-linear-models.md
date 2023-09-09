---
layout: post
title: "Generalized linear models"
categories: course/intro/
tags: /generalized-linear-models/
---

Normal model allows you to fit data belonging to the entire real domain,
but you will face situations where you want to put some additional constrain
to your model.
If you are dealing with binary data, with count data or with probabilities,
then the ordinary linear regression may not be appropriate, as your model
would allow for values which are outside from the mathematical domain of your
data.

Generalized Linear Models (GLMs for short) simply use an appropriate link
function to map the output of your linear model to the domain you choose.

For those who want a deeper dive into this kind of model,
I reccomend the McCullagh Nelder textbook, freely available [here](https://www.utstat.toronto.edu/~brunner/oldclass/2201s11/readings/glmbook.pdf).

## The logistic model and the Challenger disaster
The logistic model can be used to estimate the probability of a dichotomous
variable, namely a variable which can take two possible values:
1 (success) or 0 (failure).
As we have seen in a previous example, when we model a dichotomous variable we
can use the Binomial distribution, whose parameter must belong
to the $[0,1]$ interval.
In the logistic model we use the logistic function:

$$
f(x) = \frac{e^x}{1-e^x}
$$

to map the output of a linear regression to the $[0,1]$ interval.

![The logistic function](/docs/assets/images/glm/logistic/logistic.png)
_The logistic function_

We will roughly follow [these](https://bookdown.org/theodds/StatModelingNotes/generalized-linear-models.html)
notes, and we will apply the logistic regression to the so-called Challenger O-ring dataset.
The notes reproduce [this](https://www.jstor.org/stable/2290069) article by Dalal _et al._, where the author used the data collected before the Challenger
disaster to examine whether it would have been possible to predict the Challenger disaster before it happened.

On January 28 1986 the shuttle broke during the launch, killing several people.
The USA president formed a commission to investigate on the causes of the incident,
and one of the member of the commission was the famous physicist Richard Feynman.
NASA officials claimed that the chance of failure of the shuttle was about 1 in 100000,
while Feynman estimated that this number was actually closer to 1 in 100.
He also learned that rubber used to seal the solid rocket booster joints using O-rings,
failed to expand when the temperature was at or below 32 degrees F (0 degrees C).
The temperature at the time of the Challenger liftoff was 32 degrees F.

Feynman proved that the incident was caused by a loss of fuel due to the low temperature
[Wikipedia page](https://en.wikipedia.org/wiki/Space_Shuttle_Challenger_disaster))

Here we will take the data on the number of O-rings damaged in each mission of the 
challenger and we will provide an estimate on the probability that one O-ring becomes
damaged as a function of the temperature.
The original data can be found [here](https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring).

The logistic model is already implemented into PyMC,
but to see how it works we will implement it from scratch.

```python
import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
from matplotlib import pyplot as plt
import pymc.sampling_jax as pmjax
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")

rng = np.random.default_rng(42)

df_oring_challenger = pd.DataFrame.from_dict({
    "temperature": [53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 81],
    "damaged": [5, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "undamaged": [1, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5, 6, 5, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6]
    })
```

In the above dataset there are collected, for a set of Challenger launches,
the recorded temperature in Fahrenheit degrees, together with the number of damaged
and undamaged O-rings.
Let us rearrange the dataset as follows:

```python
df_oring = df_oring_challenger.groupby('temperature')[['damaged', 'undamaged']].apply(sum).reset_index()
df_oring['count'] = df_oring['damaged'] + df_oring['undamaged']
df_oring.head()
```

|    |   temperature |   damaged |   undamaged |   count |
|---:|--------------:|----------:|------------:|--------:|
|  0 |            53 |         5 |           1 |       6 |
|  1 |            57 |         1 |           5 |       6 |
|  2 |            58 |         1 |           5 |       6 |
|  3 |            63 |         1 |           5 |       6 |
|  4 |            66 |         0 |           6 |       6 |

We can now build our model as follows:

```python
with pm.Model() as logistic:
    alpha = pm.Normal('alpha', mu=0, sigma=20)
    beta = pm.Normal('beta', mu=0, sigma=5)
    log_theta = pm.Deterministic('log_theta',alpha + beta*df_oring['temperature'])
    theta = pm.Deterministic('theta', pm.math.exp(log_theta)/(1+pm.math.exp(log_theta)))
    y = pm.Binomial('y', p=theta, n=df_oring['count'], observed=df_oring['undamaged'])
```

For those who don't like math too much, let us show the Bayesian network
associated to the model:

```
pm.model_to_graphviz(logistic)
```
![The logistic model](/docs/assets/images/glm/logistic/model.svg)

```python
with logistic:
    trace_logistic = pmjax.sample_numpyro_nuts(draws=20000, tune=20000, chains=4,
                      return_inferencedata=True, random_seed=rng)
```

Here we used the powerful numpyro sampler to run four chains, each composed by
20000 warm up draws and 20000 remaining draws.
This sampler is much faster then the ordinary PyMC sampler, since it pre-compiles 
the code.

```
az.plot_trace(trace_logistic, var_names=['alpha', 'beta'])
fig = plt.gcf()
fig.tight_layout()
```

![Our traces](/docs/assets/images/glm/logistic/trace.png)

The traces looks very clean, but let us take a look at the trace summary

```python
az.summary(trace_logistic, var_names=['alpha', 'beta'])
```

|       |    mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:------|--------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| alpha | -11.868 | 3.342 |  -18.24  |    -5.658 |       0.033 |     0.023 |      10494 |      11515 |       1 |
| beta  |   0.221 | 0.054 |    0.119 |     0.322 |       0.001 |     0     |      10502 |      11476 |       1 |

Let us now plot our estimate for the O-ring failure probability

```python
x = np.arange(30, 80, 1)
z = trace_logistic.posterior['alpha'].values.reshape((1, 80000))*np.ones(
    len(temp)).reshape(len(x), 1) + trace_logistic.posterior['beta'
    ].values.reshape((1, 80000))*temp.reshape((len(x), 1))
prob = np.exp(z)/(1+np.exp(z))

prob_mean = np.mean(prob, axis=1)
prob_975 = np.quantile(prob, 0.975, axis=1)
prob_025 = np.quantile(prob, 0.025, axis=1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(x, 1-prob_025, 1-prob_975, color='grey', alpha=0.5,
                label='95% CI')
ax.plot(x, 1-prob_mean, color='k', label='mean probability')
ax.scatter(df_oring['temperature'], df_oring['damaged']/df_oring['count'],
           marker='x', label='raw data estimate')
legend=fig.legend(loc='upper right', framealpha=1)
ax.set_xlabel(r"T $^0F$")
ax.set_ylabel(r"P(damaged)")
fig.tight_layout()
```

![Our final estimate](/docs/assets/images/glm/logistic/probability.png)

We are slightly too optimistic at low temperature, since the lowest point
is outside from the 95% CI, but let us trust for a moment to our model.

The previous plot is quite hard to understand, as it is not clear the exact
value of the probability when $y$ is close to its boundaries.
A more easily interpretable quantity is given by the odds ratio:


$$ OR = \frac{p}{1-p}$$

The odds ratio represents how much you should bet on one result with respect on the other.
We will plot it in a log scale in order to make the plot more readable,
and will plot the odds ratio for the $y=0$ outcome, which is the inverse
of the standard $y=1$ odds ratio.

```python
odds_ratio = prob/(1-prob)
or_mean = np.mean(odds_ratio, axis=1)
or_975 = np.quantile(odds_ratio, 0.975, axis=1)
or_025 = np.quantile(odds_ratio, 0.025, axis=1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(temp, or_025, or_975, color='grey', alpha=0.5,
                label='95% CI')
ax.plot(temp, or_mean, color='k', label='Mean')
ax.set_yscale('log')
legend=fig.legend(loc='upper center', framealpha=1)
ax.set_xlabel(r"T $^0F$")
ax.set_ylabel(r"Odds Ratio (damaged)")
fig.tight_layout()
```

![The odds ratio](/docs/assets/images/glm/logistic/odds_ratio.png)

Since the odds ratio at 30 Fahrenheit degrees is close to $1000$ we have that
the failure probability, for a single O-ring, is roughly 1000 times the probability
that the O-ring will not be damaged: we can be almost sure that the O-ring will brake.

Of course, only one failure in not sufficient to have an incident. What is the probability that we have the simultaneous failure of all six O-ring at 0 Celsius degrees
(so 32 Fahrenheit degrees)?

```python
with logistic:
    theta_1 = np.exp(alpha + 32*beta)/(1+np.exp(alpha + 32*beta))
    y1 = pm.Binomial('y1', p=theta_1, n=6)
    ppc_6_32 = pm.sample_posterior_predictive(trace_logistic, var_names=['y1'])

h = [(ppc_6_32.posterior_predictive['y1'].values.reshape(-1)==k).astype(int) for k in range(7)]
sns.barplot(h)
```

![The probability mass function at 32 degrees](/docs/assets/images/glm/logistic/p_32_6.png)

As we can see, there is a little chance (less than 10%) that more than one O-ring remains undamaged.
