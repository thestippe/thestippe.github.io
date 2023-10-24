---
layout: post
title: "Categorical data"
categories: course/intro/
tags: /categorical/
image: "/docs/assets/images/multinomial/survey1.jpg"
description: "When two categories are not enough"
---

In a [previous post](/count/) we saw how we can deal for categorical data when we have two categories.
In this post we will see how to extend the previous model when we have three or more categories
by using the **multinomial** distribution, which is the most generic distribution for a finite number of discrete categories,
and has as pmf:

$$
p(x | p) = \frac{n!}{x_1!x_2!...x_k!} p_1^{x_1} p_2^{x_2}...p_k^{k_k}
$$

where $k$ is the number of categories, 

$$n = \sum_{i=1}^k x_k$$

and $p_i$ represents the probability that an object belongs to category $i$, so

$$\sum_{i=1}^k p_k =1 \,.$$

As a prior distribution one can use the **Dirichlet** distribution, which is the generalization of the Beta distribution:

$$
p(x | \alpha) = \frac{\Gamma(\alpha)}{\Gamma(\alpha_1)\Gamma(\alpha_2)...\Gamma(\alpha_k)} x_1^{\alpha_1-1} x_2^{\alpha_2-1}... x_k^{\alpha_k-1}
$$

where
$$
\alpha = \sum_{i=1}^n \alpha_i
$$
and $x$ is a vector belonging to the $p$ dimensional unit simplex:


$$
x_i > 0
$$

and 

$$
\sum_{i=1}^k x_i = 1\,.
$$

When $k=2$ we have that the Multinomial distribution corresponds to the Binomial distribution,
while the Dirichlet distribution corresponds to the Beta distribution.

![The Dirichlet distribution](/docs/assets/images/multinomial/Dirichlet.png) The Dirichlet distribution

![The Dirichlet distribution for different parameters](/docs/assets/images/multinomial/Dirichlet1.png) The Dirichlet distribution

It happened to me that, during a conference, we used a survey to understand how people got informed about the event.
Less than 50% of the audience answered to the survey, so I wanted to use a Multinomial-Dirichlet model to have an estimate about the uncertainties.

|    | channel   |   number |
|---:|:----------:|:---------:|
|  0 | friends   |       10 |
|  1 | flyier    |        2 |
|  2 | social    |        5 |
|    | **total**   |       17 |

```python

import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
from matplotlib import pyplot as plt
import pymc.sampling_jax as pmjax

plt.style.use("seaborn-v0_8-darkgrid")

rng = np.random.default_rng(42)

df_conf = pd.DataFrame({'channel': ['friends', 'flyier', 'social'], 'number': [10, 2, 5]})

n = df_conf['number'].sum()

with pm.Model() as multinomial:
    theta = pm.Dirichlet('theta', a=[1]*len(df_conf))
    y = pm.Multinomial('y', p=theta, n=n, observed=df_conf['number'])
    trace_multi = pm.sample(draws=2000, chains=4, tune=2000, random_seed=rng)

az.plot_trace(trace_multi)
```

![The trace of our model](/docs/assets/images/multinomial/dirmulti_trace.png)

Our trace looks good, so let us take a closer look to our parameter estimate:

![Our estimates for the probabilities](/docs/assets/images/multinomial/forest.png)

Of course, we do expect a high (negative) correlation between the coefficients, as the three components of $\theta$ must sum up to one,
so in order to better understand the posterior it is more appropriate to use a kernel density estimate:

![The KDE plot of the posterior](/docs/assets/images/multinomial/kde_plot.png)

```python
az.summary(trace_multi)
```

|          |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:---------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| theta[0] |  0.549 | 0.11  |    0.341 |     0.748 |       0.002 |     0.001 |       4271 |       4599 |       1 |
| theta[1] |  0.151 | 0.081 |    0.026 |     0.303 |       0.001 |     0.001 |       3254 |       3801 |       1 |
| theta[2] |  0.3   | 0.099 |    0.124 |     0.49  |       0.001 |     0.001 |       7308 |       6200 |       1 |

Thus, assuming that our sample is representative of our population (those who came to our conference)
and that people honestly filled the form we can say that it's likely that
more than the 50% of the people got informed about the event thanks to some friend, while roughly the 30% 
of the audience read about the event on some social.
Another issue of this model is that we are assuming that each answer is independent on the other answers,
while usually people goes to this kind of event in groups, but for my purpose it was sufficient
to have a rough idea of the underlying distribution.

You could be suspicious about the fact that the distribution for the flyer is so skewed.
In this case I reccomend you to perform a sensitivity analysis: try with $\alpha = (2, 2, 2)$,
as you will see the result won't change much.

Notice that our sample is quite small, so any frequentist approach based on a
large-sample approximation would be simply wrong.
However, since we are doing Bayesian statistics, we don't have to bother about
the sample size in order for our analysis to be internally consistent
(while of course any attempt to generalize
the result of the model from our sample to the entire population
should be carefully discussed).

## Conclusions and take-home message
- Each time you face a new problem, think about the constraints you should put to your model. You should always implement all the relevant constraints in order not to allow for too much freedom.
- Building a model is just one step in making inference. Think about the limitations of your model, and what are the major problems in extending your results from the sample to the population.
- By using Bayesian statistics, you can draw robust conclusions even if you have small sample.

We will now introduce a slightly more theoretical topic, namely
[conjugate models](/conjugate/),
and we will see how we can leverage them
to constrain our models.
