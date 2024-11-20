---
layout: post
title: "Introduction to the Bayesian workflow"
categories: /statistics/
subcategory: "Bayesian workflow"
tags: /bayes_workflow/
date: "2024-09-11"
section: 0
# image: "/docs/assets/images/perception/eye.jpg"
description: "How to make bayesian inference in practice"
---

While making Bayesian inference for simple problems is straightforward,
handling real word problems can be very challenging.
In order to simplify it, statisticians came out with what is known as the
**Bayesian workflow**, which is a set of rules to follow in order to
properly doing Bayesian inference.

<br>

> I have regarded concepts and techniques as guides rather than rigid recipes.
> 
> M. S. Bartlett

<br>

The Bayesian workflow is illustrated in [this paper by Gelman et al.](http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf),
and we will illustrate in this post and in the following ones how to implement
it in PyMC.

We will assume that you properly collected some data, and you want to analyze it.

## Model selection

The first step is the model selection, and in most cases you won't be
the first one dealing with that kind of data.
In order not to re-invent the wheel, you should look around and check if someone
else faced a similar problem. Good starting points are of course books,
but also blogs about statistics
(you will find some in the [resources page](/links)),
scientific articles (maybe consider taking a look at [arxiv](https://arxiv.org/)) as well as the [PyMC gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html) or [Cross-Validated](https://stats.stackexchange.com/).

If you didn't find anything useful, as general principle, [Ockham's razor](https://en.wikipedia.org/wiki/Occam%27s_razor) is a good guideline: start from the
simplest possible model.
There are many reasons for this:
- It will make easier to spot problems.
- The more a model is complex, the harder it will be for your software to fit it.
- Simple models are generally easier to understand (and to explain to your colleagues, clients or to the decision makers).
- It is easier to add complexity than to figure out which are the irrelevant features of your model.

Thus, as an example, before using a Students' t distribution, you should verify
if your data is well described by a normal distribution.

Model selection also includes variable transformations, and this step has more than
one purpose.
You may want to transform your variable to make it easily interpretable.
If you have a variable expressed in inches but your client is European, it is wise
to convert your data to the metric system.
You may also want to make your model easily interpretable, and
some transformation may make some of your parameter more meaningful.
Making a parameter meaningful has the additional advantage that it will
be easier for you to find out a sensible prior for it.
By a suitable transformation you may end up with a dataset which is
suitable for some particular model.
As an example, if you are dealing with a positive variable that spans
more than order of magnitude, taking its logarithm may simplify your life.
Moreover, ill-scaled problems may be harder to fit, so making your
variable of the order of 1 may reduce the computational effort and avoid numerical
issues.

Another fundamental aspect of the model selection is the choice of the prior,
and this is probably one of the hardest problems in Bayesian inference.
The prior choice should be generous enough not to over-constrain your
parameter, but restrictive enough to regularize your model.
The next step is dedicated to find out if your prior selection
is meaningful.

## Prior predictive checks

Your model will contain priors, but it may happen that you don't
have enough domain-specific
knowledge in order to know a priori if your guess is good enough,
so a very important step in the Bayesian workflow is to perform the
prior predictive check.

This is a very easy task, and it won't be time-consuming,
but it allows you to check if the hyperparameters in our model
are able to include our data. 

In other words, if the model predicts the outcome variable $Y$ in the range $[-10, 10]$ 
in the 95% of the simulations, but the true data are outside of this range, than you
should consider changing the hyperparameters.
As a rule of thumb, at least the 50% of the data should fall in the 50% highest density region of our prior predictive sample.

## Sampling the posterior

Now you can finally run your simulation. At the beginning, you shouldn't waste
too much time in running very large simulations, but you should rather limit
yourself to small samples (say one or two thousand of draws).
In this way you will be able to figure out in a shorter time if there's any issue.

Only once everything looks good you can draw the final sample,
and Nature recommends at least 10000 draws in order to have a sufficiently large one,
and distributing those draws in four chains is usually enough. 
Gelman suggests to only keep the last half of each of your sample,
since it is the best compromise between time and precision.
While this makes a lot of sense for very large problems, however,
in my experience for smaller problems PyMC usually only needs few thousands of iteration
to warm up.
However, I am not Andrew Gelman, so you'd better keep in mind his suggestion.

## Convergence assessment

Now you've got your trace, and it's time to verify if there was any issue with it.
We already discussed some of the tools that Arviz provides to figure
out if there's any issue. In a dedicated post we will see how to interpret
these tools and which are some possible solutions.

In some case you will simply need to run a longer chain or a longer warm-up
phase, but in some other case you will be forced to modify your model
or to only run your simulation on a subset of your data.

[This post](/statistics/trace_inspection) treats this topic more
in detail.

## Posterior predictive checks

Once everything is good you can (and should) perform the posterior predictive checks,
and they will allow you to make sure that your model reproduces
the salient features of the data.

As previously stated, all models are wrong, so it's unlikely that you will be able
to reproduce all the features of the data, 
but we should at least be able to reproduce the relevant ones,
whereby relevant I mean with respect to your questions.

Prior and posterior predictive checks
are discussed in [this post](/statistics/predictive_checks).

## Model comparison

In most cases you won't be dealing with only one model,
but you will be comparing more than one model to see which feature
is better reproduced by each model.
This part of the flow is called model comparison or model averaging,
although in most cases you won't be really averaging over the models.

If the model looks good enough, you can stick here and use the information
that you extracted from the model.

We also include sensitivity analysis into this step, as different
hyperparameters imply different models.
In this phase you want to understand how do your conclusions
change by slightly changing the priors.
If your conclusions are change a lot, then your model is not very trustful,
so you should consider changing your model or looking for new data.

Of course, it may happen that you get more data,
and that with the new data you realize that you are unable to catch some
relevant feature.
In this case, you should go back to the first step, and start again.

## Conclusions

We discussed the structure of the Bayesian workflow, in the next posts
we will discuss each step in detail with some practical example.


## Recommended readings
- <cite>Gelman et al. (2020).Bayesian Workflow.
</cite>
