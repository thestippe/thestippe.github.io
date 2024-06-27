---
layout: post
title: "A reminder on frequentist statistics"
categories: /statistics/
subcategory: Introduction
tags: /statistics_overview/
date: "2024-01-04"
# image: "/docs/assets/images/perception/eye.jpg"
description: "What the classical paradigm tells us"
section: 0
---

Before moving to Bayesian inference, let us briefly recall some basic concept of frequentist statistics.
Let us assume that we are running an experiment where the outcome $y$ is normally distributed
with mean $\mu=2$ and $\sigma=1\,.$ Let us also assume that the experimenter doesn't know
nether $\mu$ nor $\sigma\,,$ and he is interested in $\mu\,,$ so he runs an experiment where
he measures $y$ 40 times.

Let us simulate this with python

```python
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(543)  # for reproducibility

mu_true = 2
sigma_true = 1
n_obs = 40

y_obs = rng.normal(loc=mu_true, scale = sigma_true, size=n_obs)
```

He can easily compute a point estimate[^1] for both $\mu$ and $\sigma\,,$
and he gets

[^1]: We recall that estimating is the procedure of associating a value (the estimate) to a parameter (the estimand), and this is done by means of a statistic (any function of the sample).

```python
mu_est = np.mean(y_obs)
sigma_est = np.std(y_obs)
```

```python
mu_est
```

<div class="code">
1.9161258515048432
</div>

```python
sigma_est
```

<div class="code">
0.8076823501920702
</div>

Those are only point estimate, but they don't tell us anything about how far we are from the true values.
Since he is interested in $\mu\,,$ he computes a Confidence Interval for it, and he does so by using
the Central Limit theorem, which states that, in the long run, the observed sample mean $\bar{Y}_n$ obeys

$$
\sqrt{n}(\bar{Y}_n - \mu) \sim \mathcal{N}(0, \sigma^2)
$$

where $\mu$ and $\sigma$ are the true mean and standard deviation respectively.
Of course, he doesn't know nether $\mu$ nor $\sigma\,,$ but he can use his point estimates for them.
He then uses these values to estimate the $95%$ two tail CI as follows:

```python
ci = [mu_est + norm.ppf(0.025)*sigma_est/np.sqrt(len(y_obs)),
      mu_est + norm.ppf(0.975)*sigma_est/np.sqrt(len(y_obs))]

ci
```
<div class="code">
[1.665827097340284, 2.1664246056694023]
</div>

The main issue with CI is that it is easily misinterpreted as the probability for the parameter
$\mu$ to fall within the observed CI.
However, $\mu$ is not a random variable, but it is a number.
We therefore have either probability 1 or 0 that the true value of $\mu$ will fall within the observed CI
if $\mu$ is inside the CI or not respectively.

We have that, if we repeat the experiment, we have 0.95 probability that the new observed CI
will include $\mu\,.$

```python
fig = plt.figure()
n = 0
n_exp = 200
ax = fig.add_subplot(111)
for k in range(n_exp):
    y_obs_new = rng.normal(loc=mu_true, scale = sigma_true, size=n_obs)
    mu_est_new = np.mean(y_obs_new)
    sigma_est_new = np.std(y_obs_new)
    ci_new = [mu_est_new + norm.ppf(0.025)*sigma_est_new/np.sqrt(len(y_obs)),
              mu_est_new + norm.ppf(0.975)*sigma_est_new/np.sqrt(len(y_obs))]
    if mu_true < ci_new[1] and mu_true> ci_new[0]:
        n += 1
        ax.axvline(x=k/n_exp, ymin=ci_new[0]/4, ymax=ci_new[1]/4, color='lightgray')
    else:
        ax.axvline(x=k/n_exp, ymin=ci_new[0]/4, ymax=ci_new[1]/4, color='red')
ax.axhline(y=mu_true/4, color='k', ls=':')
ax.axhline(y=ci[0]/4, color='steelblue', ls=':')
ax.axhline(y=ci[1]/4, color='steelblue', ls=':')
ax.set_title(f"We have p={n/n_exp} that the {95}% CI includes the true mean")
ax.set_yticks(np.linspace(0, 1, 5))
ax.set_yticklabels(4*np.linspace(0, 1, 5))
ax.set_xticks([0,1])
ax.set_xticklabels([0,n_exp])
fig.tight_layout()
```

![](/docs/assets/images/statistics/frequentist/ci.webp)

We are therefore not providing an uncertainty for our parameter $\mu\,,$
but we are rather doing so for our confidence interval itself.
This is quite reasonable, since in the "classical" paradigm the parameters are fixed,
and we have no way to associate a probability to them.

## Conclusions

We have seen how to estimate parameters in statistics, together with some conceptual
difficulty of their interpretation.
Starting from the next post we will enter into the core of this section, namely Bayesian statistics.