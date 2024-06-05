---
layout: post
title: "Trace inspection"
categories: /statistics/
subcategory: "Bayesian workflow"
tags: /trace_inspection/
date: "2024-01-21"
section: 1
# image: "/docs/assets/images/perception/eye.jpg"
description: "Finding issues in MCMC convergence"
---

In the last post we introduced the concept of Bayesian workflow.
Starting from this post, we will discuss some of the most important aspects,
and we will do so by starting with the **trace assessment**.

In this post I will not provide you the details of the computation,
as our only purpose will be to look for pathologies into the trace.
I will often exacerbate some issue to make the issues visible,
and I will do this by means of really short traces or crazy parameters.

Tricking the NUTS sampler is not easy, but this may happen when
your parameter space is large or has some bad structure.
In this case, however, the issues overlap and it is less clear how
to handle them. For this reason I preferred to switch to simpler models
but with bad parameter choices.
I also used on purpose a quite small number of draws, as a larger one would
make harder to spot by eye the pathologies.

## Convergence not reached

The first kind of issue is usually very easy to spot, and it's the non-stationary
trace. In this case we didn't left enough time to the sampler to reach the
stationary regime, and in this case the trace shows a clear trend.

The solution, in this case, is quite clear: you should increase the number of tuning
draws.

![Convergence not reached](/docs/assets/images/statistics/trace/early.webp)

As we have already seen, the above plot can
be done with Arviz' [plot_trace function](https://python.arviz.org/en/stable/api/generated/arviz.plot_trace.html).

## Large autocorrelation

Let us take a look at a second trace.

![Large autocorrelation trace](/docs/assets/images/statistics/trace/acorr_trace.webp)

In this case the trace does not show any trend. There is however a very large autocorrelation.
This can be seen on the right hand side of the plot, by observing that
the trace is *globally* stationary, but *locally* the average is not
constant (take a look at the $v$ variable around $i=150$, it is clearly
visible some kind of bump).

![Autocorrelation Function (ACF) plot for large autocorrelation](/docs/assets/images/statistics/trace/acorr_acorr.webp)

You can plot the autocorrelation by using
the [plot_autocorr function](https://python.arviz.org/en/stable/api/generated/arviz.plot_trace.html).


This kind of issue becomes clear when one plots the autocorrelation function
with arviz.
We can clearly see an oscillating behavior, as well as a very large asymptotic
estimate for the upper bound of the coefficients.

Also the $\hat{R}$ statistics can be helpful in spotting this kind
of issue, as in these cases it generally differs from 1.

|       |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| v     |  2.193 | 0.389 |    1.531 |     2.921 |       0.024 |     0.017 |        268 |        520 |    1.02 |
| sigma |  0.662 | 0.103 |    0.506 |     0.845 |       0.008 |     0.006 |        238 |        290 |    1.03 |

We recall that the above table can be obtained
by using the [summary function](https://python.arviz.org/en/stable/api/generated/arviz.summary.html).

While in these cases the core part of the distribution is reliable enough,
so you can safely estimate the mean, you should never trust to estimates
involving peripheral regions of the posterior, like the $95\%$ HDI,
unless your sample size is large enough.

In this case one should first try to leave to the sampler more
time to find the optimal parameters.
The NUTS sampler, by construction, tries to reduce the autocorrelation
as much as possible in the tuning phase.

This may however be very difficult due some pathologies of the posterior distribution.
As an example, this may happen if one parameter has a large posterior for very large
values while for another one the optimal region is for very small values.
This kind of problems becomes even worst when there is a large correlation
between the parameters.

You should therefore try and re-parametrize your model. A simple rescaling in
the parameters may be sufficient, but sometimes it is necessary to find
a parametrization where the parameters are decoupled.

As an example, if your likelihood reads

$$
Y \sim \mathcal{N}(\mu, \sigma)
$$

you should consider rewriting it as

$$
Y \sim \mu + \mathcal{N}(0, \sigma)
$$

or you could even introduce an auxiliary random variable $X \sim \mathcal{N}(0, 1)$
and rewriting your model as

$$
Y \sim \mu + \sigma X
$$

Another possible solution is to use a smaller dataset, or to think
if you are constraining too much or too little your parameters 
and change your prior accordingly.

## The rank plot

Here we will deal with the same kind of issues that we have discussed above,
but where the presence of the issue cannot be clearly seen with the above tools.

![A trace plot which looks almost good](/docs/assets/images/statistics/trace/acorr_large_trace.webp)

The above trace may look OK at a first glance, there is not clear autocorrelation
pattern and the chains are quite similar.
They are however not identical, and this should warn you.

![An ACF plot which looks good](/docs/assets/images/statistics/trace/acorr_large_acorr.webp)

Also by looking at the autocorrelation plot it may be unclear if there is
any issue.


|       |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| v     |  2.195 | 0.388 |    1.539 |     2.927 |       0.008 |     0.006 |       2162 |       2807 |       1 |
| sigma |  0.659 | 0.099 |    0.512 |     0.851 |       0.002 |     0.002 |       2307 |       2803 |       1 |

From the trace summary you would hardly guess that there is some issue,
as the effective sample size is above 2000 and the $\hat{R}$ is one.

However, by looking at the [rank plot](https://python.arviz.org/en/stable/api/generated/arviz.plot_rank.html), you can easily realize that there is
something going on.

![The rank plot for the above trace](/docs/assets/images/statistics/trace/acorr_large_rank.webp)

In the above figure, the chain number $1$ of the variable $v$
shows some clear departure from the uniform distribution around $2000$,
and the same happens for the second bar of the trace number $0\,.$

If you want a reliable estimate of the entire distribution,
you should make sure that this kind of pattern is not visible.
If this is the only issue, it is likely that a larger number of draws is sufficient.

## Jumping traces

In some cases your model may be ill-defined, and it might happen that two subsets of parameters with different values can be exchanged without changing
the probability.
In these cases what might happen is that different chains of a parameter
will converge to different values, as the case below.

![The trace of a pathological model](/docs/assets/images/statistics/trace/superposition.webp)

In order to better see this issue, let us take a look at the [forest plot](https://python.arviz.org/en/stable/api/generated/arviz.plot_forest.html) of
the model.

![The forest plot of the model above](/docs/assets/images/statistics/trace/superposition_forest.webp)

The estimate for $\mu_0$ from the 0-th chain is above 4, while the remaining
chains are below 2.
When we combine the four chains, of course, the estimate
is totally unreliable

|       |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| mu[0] |  2.186 | 1.256 |    1.34  |     4.444 |       0.624 |     0.478 |          7 |         29 |    1.53 |
| mu[1] |  3.649 | 1.261 |    1.399 |     4.539 |       0.626 |     0.479 |          7 |         29 |    1.53 |
| mu[2] |  5.66  | 0.123 |    5.438 |     5.887 |       0.003 |     0.002 |       2065 |       2125 |    1    |

This can be clearly seen by the crazy value of $\hat{R}\,,$
since the estimated variance from the combined chain is very different from
the one estimated by combining the variances of the single chains.

In this case, the only safe solution is to re-parametrize your model,
imposing an order to the parameters.

## Conclusions

We have seen some of the most common kind of issues that you may encounter, some tools to diagnose these
issues and some possible solutions.
This is of course only a selected list, and you may encounter some other
issues, especially when the model complexity grows.
You should therefore always carefully inspect the trace to exclude
issues which may affect your conclusions.
