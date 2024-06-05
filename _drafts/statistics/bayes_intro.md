---
layout: post
title: "Introduction to Bayesian inference"
categories: /statistics/
subcategory: Introduction
tags: /bayes_intro/
date: "2024-01-05"
# image: "/docs/assets/images/perception/eye.jpg"
description: "A little bit more about Bayesian inference"
section: 0
---

The main topic of this section will be statistics, and we will mostly discuss Bayesian
models.

## Why should you learn Bayesian statistics

Bayesian statistics attracted much interest in the last years, and in the last decades
many Bayesian models have been developed to tackle any kind of problem.

![The number of Nature research article 
matching the keyword "bayes"](/docs/assets/images/statistics/intro/nature_count.webp)

Bayesian statistics can be a valuable tool for any data scientist, as it easily allows you
to build, fit and criticize any kind of model with a minimal effort.
In Bayesian inference you don't only get an estimate for your parameters,
but you get the entire probability distribution for them, and this implies that
you can immediately get the uncertainty for your parameters.

In any statistical model you must specify a likelihood, which represents
the probability distribution which generated the data, and we will refer to this quantity
as

$$ P(X | \theta)$$

where $\theta$ represents the unknown parameter vector.
In a Bayesian model you must moreover specify a prior distribution for the parameter
set 

$$P(\theta)\,.$$

You can therefore compute the probability distribution of your parameters given the data
by using the Bayes theorem:

$$
P(\theta | X) = \frac{P(X | \theta)}{P(X)} P(\theta)
$$

Since the denominator $$P(X)$$ is a normalization constant independent on $\theta$ which can be computed by normalizing the left hand side of the equation
to 1, its dependence is usually neglected and the Bayes theorem is usually rewritten as

$$
P(\theta | X) \propto P(X | \theta) P(\theta)\,.
$$

## A historical tour in MCMC

While the normalization constant $$P(X)$$ can be **in principe** computed for any model,
it can be analytically computed only for a very limited range of models, namely
the conjugate models, and for this
reason Bayesian models have not been so popular for a long time.
However, thanks to the development of Monte Carlo sampling techniques, it became possible
to easily sample (pseudo) random numbers according to any probability distribution.
These techniques have been developed during the WWII by the Manhattan project
group, and it soon became popular among scientists to perform numerical simulations.
Up to few years ago, this was however only possible for people with a strong background
in numerical techniques, as it was necessary to implement from scratch the sampler.
Nowadays things have changed, and Probabilistic Programming Languages
such as PyMC, STAN or JAGS allows anyone to implement and fit any
kind of model with a minimal effort.

## Some philosophical consideration
You may have heard about the war of statistics, which is a debate which lasted
almost one century between frequentist statisticians and Bayesian ones.
At the beginning of the last century, a group of statisticians tried and promote
Bayesian statistics as the only meaningful way to do statistical inference.
According to them, the prior should have encoded all the available information
together with any personal consideration of the researcher. In this way,
the posterior probability distribution $P(\theta | X)$ can be interpreted
as the updated version of the researcher's beliefs once the observed data $X$
has been taken into account by a perfectly rational person.
This philosophy has been rejected by the majority of the statisticians,
as they considered this "subjective" probability meaningless.

Today, however, the prior probability is only considered a regularization tool,
which allows you to use Bayes theorem to compute the posterior probability [^3].
The results should only weakly depend on the prior choice, and this dependence
must be taken into account when reporting the fit results.
In this way, since the [Bernstein–von Mises](https://en.wikipedia.org/wiki/Bernstein%E2%80%93von_Mises_theorem) theorem ensures you that
in the long run the Bayesian inference will produce the same results of the model
with the same likelihood, one can stick to the usual frequentist interpretation.


<div class="emphbox">
Nowadays Bayesian inference is accepted by most statisticians,
and an entire workflow has been developed to ensure that the inference procedure
has been properly performed.
</div>

## Bayesian inference as a tool in the replication crisis

During the beginning of the 2010s, scientists realized that a large part
of research articles were impossible to replicate.
In these years, many unsound scientific results have been found, and claims such as
[the existence of paranormal phenomena](
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4706048/) or that 
[a dead salmon was able to recognize human emotions](
http://cda.psych.uiuc.edu/multivariate_fall_2013/salmon_fmri.pdf) were not as rare
as one would expect [^1].

[^1]: One is a joke, the other is not. Choosing which one is a joke is left to the reader as an exercise.

A new research field emerged in these years, namely meta-science (which means the science
which studies science), and this produced many proposals to tackle the scientific crisis.
One of the issues that came out was that [the "golden standard" tool of the 0.05 statistical
significance was often misused or misinterpreted](https://www.nature.com/articles/d41586-019-00874-8),
and statisticians suggested that using a broader set of tools to perform statistical
analysis would have reduced this problem.
As you might imagine, due to the simple interpretation and of
the possibility to easily implement structured models and of combining
different data sources,
Bayesian inference has been popularized by statisticians as one of these tools,
and it has gained a lot of attention by the scientific community.

## Technical considerations of using Bayesian inference

There are also technical considerations which one should take into account
when choosing the analysis method.
For simple models there is no reason why one should prefer Bayesian inference
to frequentist-based ones,
at least as long as the sample size is large enough
to allow you using the central limit theorem.
However, for complex models, things soon change. One may naively be tempted
to simply use the maximum likelihood estimation, which is in principle a
valid way to drop out the prior dependence. However, this method relies
on finding the extreme of a function, which implies maximizing the derivative
of the likelihood. This approach soon becomes unstable as soon as the model
complexity grows, and you should spend a lot of effort in ensuring that
your numerical approximation is close enough to the true maximum.
For these reasons, statistical textbooks devoted to complex models such as
longitudinal data analysis models provide many different methods
to fit the frequentist models [^2].

[^2]: Even if you analytically compute the derivative,
you must find its zeros, and there is no stable procedure to do so for higher dimensional problems.

The Bayesian method, on the other hand, does not require to compute any
derivative, as you simply need to sample the posterior distribution and, as
we will see, it is very easy to assess the goodness of such a sampling procedure.
If you then want to compute any point estimate  $$\mathbb{E}[f(\theta)] $$, what you have to do is to compute
the corresponding expectation value on the sample $$\left\{\theta_i\right\}_i:$$

$$
\mathbb{E}[f(\theta)] \approx \frac{1}{N} \sum_{i=1}^N f(\theta_i)
$$

and, as we will discuss, the estimate of the error associated with this procedure is already
implemented in the sampling engine.

## Problems you will likely face if you use Bayesian inference

There is of course no free lunch, as every method has pros and cons.

First of all, unless you are trained as a statistician,
at the beginning you will
likely face one problem: the way of thinking.
Probabilistic thinking requires a training, but in the long run
it will be a priceless tool, as probability is the **only** consistent
way to quantify uncertainties.

From a more practical point of view, the main drawback of using Bayesian methods is that sampling may require time,
and while having a high quality sample with few parameters
requires seconds, you may need hours or more for a good sample if you are
dealing with thousands of parameters.

You should also consider that the prior selection might take require
some effort, especially when you are dealing with a new problem
or with a new kind of model.

## Conclusions

I hope I convinced you that Bayesian inference is a valuable instrument in the toolbox of
any data scientist.
In the next articles I will show you how to implement, check and discuss Bayesian models
in Python.
As usual, if you have criticisms or suggestions, feel free to write me.

In this section of the blog we will both discuss Bayesian inference and
the Bayesian workflow.
I will use Python to perform the computation, and I will use the PyMC ecosystem.

[^3]: Here we distinguish between Bayesian inference, which is a set of mathematical tools, and Bayesian inference, which we identify as the subjective interpretation of probability.