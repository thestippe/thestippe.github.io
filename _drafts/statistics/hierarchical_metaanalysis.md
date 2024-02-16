---
layout: post
title: "Hierarchical models and meta-analysis"
categories: /statistics/
subgategory: "Advanced models"
tags: /hierarcical_metaanalysis/
date: "2024-01-30"
# image: "/docs/assets/images/perception/eye.jpg"
description: "How hierarchical models can be used to analyze scientific literature"
section: 1
---

In the last post we discussed how to build a hierarchical model.
These models are often used in meta-analysis and reviews,
*i.e.* in academic publications where the results of many studies are collected,
criticized and combined together.
In this kind of study using a full pooling would not be appropriate,
as each study is performed at its own conditions,
so a hierarchical model is much more appropriate to combine the results together.

We will use this method to re-analyze an old meta-analysis by Daryl Bem,
a well known researcher who published the famous 
["Bem meta-analysis"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4706048/)
where he suggested that he found some evidence of precognition.
This paper became very famous both because of the conclusions
and because researchers failed to replicate its findings.

By analyzing the methodological flaws used in that paper,
other scientist started proposing many methods to improve
the robustness of the research results.

We won't analyze that article, since it combined many kinds of experiments,
and this makes the overall analysis more involved. We will instead use
another paper by the same author which can be found
on [this page of the CIA website](https://www.cia.gov/readingroom/docs/CIA-RDP96-00789R003200110001-4.pdf) (yes, CIA has been really interested into paranormal
activity).
We choose this article as it simply involves a binomial likelihood.
The article, in fact, summarizes the results of 11 experiments
which has equi-probable binary outcome, so by random guessing one
would expect, on average, a success percentage of the $50\%\,.$

In the analysis we will use the **Region Of Practical Equivalence** (ROPE)
to assess if the effect is practically equivalent with absent.
We will conclude that there is no evidence of precognition if the
$94\%$ Highest Density Region for the average success ratio
is entirely included in the region $[0, 0.75]\,.$
We will instead conclude that there is evidence of precognition
if the $94\%$ HDI is entirely inside the $[0.75, 1]$ region.
If the $94\%$ HDI crosses 0.75 we will conclude that the analysis is inconclusive.

The limit 0.75 seems quite strong, but since a positive finding
would contradict the current scientific knowledge, we require a strong evidence
in order to get a positive result.
Notice that the ROPE choice must be done *before* the dataset is seen,
otherwise we could tune the choice on the data.


As before, we will take

$$
\begin{align}
\alpha \sim & \mathcal{HN}(10)
\\
\beta \sim & \mathcal{HN}(10)
\\
\theta_i \sim & \mathcal{Beta}(\alpha, \beta)
\\
y_i \sim & \mathcal{Binom}(\theta_i, n_i)
\end{align}
$$

where $\mathcal{HN}(\sigma)$ denotes the Half Normal distribution.

Our dataset will consist into the 4th and 5th columns of table 1, which we provide
here

|   n |   y |
|----:|:----|
|  22 |   8 |
|   9 |   3 |
|  35 |  10 |
|  50 |  12 |
|  50 |  18 |
|  50 |  15 |
|  36 |  12 |
|  20 |  10 |
|   7 |   3 |
|  50 |  15 |
|  25 |  16 |

```python
import pandas as pd
import numpy as np
import seaborn as sns
import pymc as pm
import arviz as az
import scipy.stats as st
from matplotlib import pyplot as plt

rope_reg = [0, 0.75]

df = pd.DataFrame({"n": [22, 9, 35, 50, 50, 50, 36, 20, 7, 50, 25], 
                   "y": [8, 3, 10, 12, 18, 15, 12, 10, 3, 15, 16]})

rng = np.random.default_rng(42)

with pm.Model() as binom_meta:
    alpha = pm.HalfNormal("alpha", sigma=10)
    beta = pm.HalfNormal("beta", sigma=10)
    theta = pm.Beta('theta', alpha=alpha, beta=beta, shape=len(df['n'].values))
    y = pm.Binomial('y', p=theta, n=df['n'].values,
                  observed=df['y'].values)

with binom_meta:
    trace_meta = pm.sample(5000, tune=5000, chains=4, target_accept=0.98, random_seed=rng)

az.plot_trace(trace_meta)
```

![The trace of the hierarchical model](/docs/assets/images/statistics/hierarchical_meta/trace.webp)

```python
az.summary(trace_meta)
```

|           |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:----------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| alpha     |  8.198 | 3.069 |    2.821 |    13.908 |       0.03  |     0.021 |      10166 |      11478 |       1 |
| beta      | 14.197 | 5.294 |    5.034 |    24.155 |       0.05  |     0.035 |      10506 |      11784 |       1 |
| theta[0]  |  0.365 | 0.076 |    0.22  |     0.502 |       0     |     0     |      28685 |      14641 |       1 |
| theta[1]  |  0.356 | 0.091 |    0.19  |     0.531 |       0.001 |     0     |      26656 |      13005 |       1 |
| theta[2]  |  0.316 | 0.063 |    0.202 |     0.438 |       0     |     0     |      30418 |      13072 |       1 |
| theta[3]  |  0.278 | 0.054 |    0.179 |     0.379 |       0     |     0     |      27290 |      15091 |       1 |
| theta[4]  |  0.361 | 0.057 |    0.252 |     0.467 |       0     |     0     |      32101 |      14623 |       1 |
| theta[5]  |  0.32  | 0.057 |    0.214 |     0.426 |       0     |     0     |      29704 |      14628 |       1 |
| theta[6]  |  0.346 | 0.064 |    0.229 |     0.468 |       0     |     0     |      28892 |      14533 |       1 |
| theta[7]  |  0.431 | 0.08  |    0.283 |     0.584 |       0     |     0     |      29458 |      14330 |       1 |
| theta[8]  |  0.383 | 0.097 |    0.2   |     0.561 |       0.001 |     0     |      30103 |      13251 |       1 |
| theta[9]  |  0.32  | 0.056 |    0.214 |     0.426 |       0     |     0     |      32310 |      13805 |       1 |
| theta[10] |  0.516 | 0.08  |    0.368 |     0.664 |       0.001 |     0     |      19487 |      13738 |       1 |

There is no evident issue in the sampling procedure.

```python
az.plot_forest(trace_meta, var_names=['theta'], rope=rope_reg)
```

![The forest plot of the hierarchical model](/docs/assets/images/statistics/hierarchical_meta/forest.webp)


None of the studies suggests that there is any evidence of precognition.
We can also estimate the overall average as well as the effective sample
size.

```python
with binom_meta:
    logit_mu = pm.Deterministic("logit_mu", pm.math.log(alpha/beta))
    log_neff = pm.Deterministic("log_neff", pm.math.log(alpha+beta))

with binom_meta:
    ppc = pm.sample_posterior_predictive(trace_meta, var_names=['alpha', 'beta', 'logit_mu', 'log_neff'])

fig = plt.figure()
ax = fig.add_subplot(111)
az.plot_pair(ppc.posterior_predictive, var_names=["logit_mu", "log_neff"], kind="kde", ax=ax)
ax.set_xlim([-5, 5])
ax.set_ylim([1, 4])
fig.tight_layout()
```

![The kernel density estimate for mu and for the effective sample size of the hierarchical model](/docs/assets/images/statistics/hierarchical_meta/kde.webp)

## Conclusions

We applied the beta binomial hierarchical model to a meta-analysis on
precognition. We also introduced the Region Of Practical Equivalence (ROPE).
