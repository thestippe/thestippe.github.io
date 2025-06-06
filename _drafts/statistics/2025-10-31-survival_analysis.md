---
categories: /statistics/
date: 2025-10-31
description: Estimating waiting times
layout: post
section: 2
subcategory: Advanced models
tags: /survival_intro/
title: Introduction to survival analysis

---



There are situations where your task is to estimate the waiting time before
a certain event happens, and survival analysis is the branch of statistics
which deals with this kind of study.

In general, time can be either considered a continuous variable or a discrete
one. For the moment we will assume that it's a continuous one.

Since we are dealing with a waiting time, our variable must be non-negative.

We will focus for now on parametric models, although non-parametric models
are very popular in survival analysis.

The analyzed event can be either the time before a component fails
or the occurrence of some biological change like the infection of one patient
or even the next eruption of a volcano.

## Mathematical background

Let us consider a random variable $T$ with pdf $p$ and cumulative $F\,,$ we define the **survival function** $S$ as:

$$ F(t) = P(t\leq T) = \int_0^t p(u) du = 1-S(t)$$

We assume that at $t=0$ the event is not happened, so $S(0)=1$ while we
assume that we are certain that the event must occur, so $\lim_{t\rightarrow \infty} S(t)=0\,.$ 

We may alternatively assume that the event does not happen with probability $p_0\,,$
and in this case we may modify the above assumption with 
$\lim_{t\rightarrow \infty} S(t)=p_0\,.$ 


We define the **hazard function** as

$$h(t) = \lim_{\Delta t \rightarrow 0} \frac{P(t< T \leq t+\Delta t | T>t)}{\Delta t} = \lim_{\Delta t \rightarrow 0} \frac{P((t< T \leq t+\Delta t) \cap T>t)}{P(T>t)\Delta t} = \lim_{\Delta t \rightarrow 0} \frac{P(t < T \leq t+\Delta t)  }{\Delta t} \frac{1}{P(T>t)}  =
\frac{1}{S(t)}\lim_{\Delta t \rightarrow 0} \frac{F(t+\Delta t) - F(t)}{\Delta t }= \frac{F'(t)}{S(t)}  = \frac{f(t)}{S(t)} $$

Since $h$ is the ratio of two positive quantities, it is positive itself.

We have that

$$ h(t) =  \frac{f(t)}{S(t)} = -\frac{S'(t)}{S(t)} = -\frac{d}{dt}\log S(t) $$ 

which can be inverted by first integrating and then exponentiating:

$$ S(t) = \exp\left(-\int_0^t h(u) du\right) $$

Notice that, if we assume that $\lim_{t\rightarrow \infty} S(t)=0\,,$ we must require that $\lim_{t\rightarrow \infty}\int_0^t h(u) du = \infty\,.$
If we otherwise assume that $\lim_{t\rightarrow \infty} S(t)=p_0\,,$ we must require that $\lim_{t\rightarrow \infty}\int_0^t h(u) du = -\log\left(p_0\right)\,.$

We define the **cumulative hazard function** as

$$ H(t) = \int_0^t h(u) du $$

And it is related to the survival function by

$$ S(t) = \exp\left(-H(t)\right) $$

## Censoring

One of the main issues of survival analysis is that we are only able
to observe our system for a finite amount of time $c$, and in this
period the event may or may not occur.

Let us assume that we performed a study with duration $c\,,$ if we do not observe the event within the end of the study we cannot conclude that the event did not 
happen,
we can only conclude that it did not happen within time $c$. We assume that the event must happen at some time.
We introduce the outcome variable $y$ as $$y = \min(t, c)$$ and define the
**censoring status** variable $\delta$ which indicates if the event was observed or not

$$
\delta =
\begin{cases}
1\,\,\,  if \,\,\,  t < c \\
0 \,\,\,  if \,\,\,  t \geq c
\end{cases}
$$

and if it is not observed we say that it is **censored**.
If the event is not censored then its contribution to the likelihood is, as usual, $f(t)\,,$ but if we do not observe
the event within time $c$ then all we can conclude is that
the event happened after time $c\,,$ and the probability for
this is $S(c)\,.$
Thus the likelihood can be written as

$$L = f(y)^\delta S(y)^{1-\delta} = (h(y) S(y))^\delta S(y)^{1-\delta}= h(y)^\delta S(y)$$

and this quantity is sometimes defined as the **generalized likelihood**.

## Wrong methods for accounting of censoring

If you are new to survival analysis, and you don't know how to correctly
include censoring in your model, you may end up with a biased estimate of the 
waiting time.

Let us see why a naive handling of the unobserved data may end up with a wrong
estimate of the parameters.

Let us generate 100 fake observations, distributed according
to 

$$
Y \sim \mathcal{Exp}(1)
$$

Let us also assume that our study started at $t=0$ and ended at $t=c=1.5\,.$

```python
import numpy as np
import pymc as pm
import arviz as az
import seaborn as sns
from matplotlib import pyplot as plt

rng = np.random.default_rng(seed=123321)

w = rng.exponential(size=100)

w_cp = w.copy()

c = 1.5

censoring = (w>c).astype(int)

```

### Naive method 1: putting a threshold

A first attempt could be to replace the unobserved event with the
censoring time.

```python

w_cp[w_cp>c] = c

with pm.Model() as uncensored_model:
    lam = pm.Exponential('lam', lam=0.5)
    y = pm.Exponential('y', lam=lam, observed=w_cp)
    trace_uncensored = pm.sample(tune=5000, draws=5000, random_seed=rng)

az.plot_trace(trace_uncensored)
```

![The trace of the truncated model](/docs/assets/images/statistics/survival_intro/trace_uncensored.webp)

```python
az.summary(trace_uncensored)
```

|     |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:----|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| lam |  1.328 | 0.133 |    1.083 |     1.575 |       0.001 |     0.001 |       9164 |      12834 |       1 |

From the above summary we can observe that the $94\%$
HDI for this model does not contain the true value for the parameter.

### Naive method 2: dropping the unobserved units

Another wrong method to deal with censoring is to 
only include in our dataset units which has an observation,
while excluding the remaining.

```python
w_1 = w[w<c]

with pm.Model() as dropped_model:
    lam = pm.Exponential('lam', lam=0.5)
    y = pm.Exponential('y', lam=lam, observed=w_1)
    trace_dropped = pm.sample(tune=5000, draws=5000, random_seed=rng)

az.plot_trace(trace_dropped)
```

![The trace of the dropped model](/docs/assets/images/statistics/survival_intro/trace_dropped.webp)

```
az.summary(trace_dropped)
```

|     |   mean |   sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:----|-------:|-----:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| lam |   1.73 | 0.19 |    1.368 |     2.082 |       0.002 |     0.001 |       8492 |      14744 |       1 |

This estimate is even worse than the above one.

## Correct method

Let us now show that a correct inclusion of the censoring
into the model gives a better estimate of the average lifetime.
In PyMC, censoring can be simply implemented by using the [Censored class](https://www.pymc.io/projects/docs/en/latest/api/distributions/censored.html).

```python
with pm.Model() as censored_model:
    lam = pm.Exponential('lam', lam=0.5)
    dist = pm.Exponential.dist(lam=lam)
    y_censored = pm.Censored('y_censored', dist, observed=w_cp, upper=c, lower=None)
    trace_censored = pm.sample(tune=5000, draws=5000, random_seed=rng)

az.plot_trace(trace_censored)
```

![The trace of the censored model](/docs/assets/images/statistics/survival_intro/trace_censored.webp)

```python
az.summary(trace_censored)
```

|     |   mean |   sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:----|-------:|-----:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| lam |   1.08 | 0.12 |    0.863 |     1.314 |       0.001 |     0.001 |       9889 |      14108 |       1 |

We now have that our estimate is correct within one standard deviation, and
this is a huge improvement with respect to both the naive methods.

## Comparison of the results

In order to better understand the difference in the estimate, let us now sample
and plot the posterior predictive distributions of the three models.

```python
with censored_model:
    y_pred = pm.Exponential('y_pred', lam=lam)
    ppc_censored = pm.sample_posterior_predictive(trace_censored, var_names=['y_pred'])

with uncensored_model:
    y_pred = pm.Exponential('y_pred', lam=lam)
    ppc_uncensored = pm.sample_posterior_predictive(trace_uncensored, var_names=['y_pred'])

with dropped_model:
    y_pred = pm.Exponential('y_pred', lam=lam)
    ppc_dropped = pm.sample_posterior_predictive(trace_dropped, var_names=['y_pred'])

bins = np.arange(0, 10, 0.5)
xlim = [0, 10]
ylim = [0, 1.2]

xticks = [0, 5, 10]
yticks = [0, 0.5, 1]

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
ax1.hist(ppc_uncensored.posterior_predictive['y_pred'].values.reshape(-1), density=True, bins=bins,  alpha=0.8)
ax1.hist(w, density=True, bins=bins, alpha=0.6)
ax1.set_title('Uncensored')
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)

ax2 = fig.add_subplot(132)
ax2.hist(ppc_dropped.posterior_predictive['y_pred'].values.reshape(-1), density=True, bins=bins,  alpha=0.8)
ax2.hist(w, density=True, bins=bins, alpha=0.6)
ax2.set_title('Dropped')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)

ax3 = fig.add_subplot(133)
ax3.hist(ppc_censored.posterior_predictive['y_pred'].values.reshape(-1), density=True, bins=bins,  alpha=0.8)
ax3.hist(w, density=True, bins=bins, alpha=0.6)
ax3.set_title('Censored')
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)

fig.tight_layout()

```

![The PPC distribution for the three models](/docs/assets/images/statistics/survival_intro/ppc_compare.webp)

In the above figures, the red histogram corresponds to the true (uncensored) data, while
the blue one corresponds to the posterior predictive distribution of our model.
The effect of the bias for method 1 and 2 is quite evident, while the censored
model predicts a distribution which is quite close to the true data.

## Conclusions

We introduced survival analysis, and we introduced some main concept as
the hazard function and the survival function.
We also discussed censorship, and we showed with an example why it is important
to correctly account of censoring.
