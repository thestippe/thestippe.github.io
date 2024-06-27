---
layout: post
title: "An overview to statistics"
categories: /statistics/
subcategory: Introduction
tags: /statistics_overview/
date: "2024-01-02"
# image: "/docs/assets/images/perception/eye.jpg"
description: "What well we talk about"
section: 0
---

Before we start with the technical staff, we should clarify what well we talk about.
Since most of this section will be devoted to statistics, it is better to clarify what do we mean with 
statistics.

Most of the definitions of statistics are very similar to the following one

<br>

> Statistics is the discipline that concerns the collection, organization,
> analysis, interpretation, and presentation of data.
> 
> Wikipedia

<br>

This is a good definition, but we may also define statistics as the profession performed
by statisticians. In this case the question is: **what do statisticians do?**

One of my favourite answers to this question has been given by the
Lawrence Livermore National Laboratory statistician Kristin Lennox in [this video](https://www.youtube.com/watch?v=eDMGDhyDxuY&t=826s),
in what she defines as the "Central Dogma of Statistics"

<br>

> Statisticians use probability to quantify uncertainty.
> 
> Kristin Lennox

<br>

They do so because, in the decision-making process, it is crucial to properly account for uncertainties.
However, not everybody has been trained as a statistician[^1], so you will often find
suboptimal ways to take uncertainties into account.
I have a very strong opinion, which is the following:

[^1]: I have not!

<div class="emphbox">
Everybody who uses data, not only statisticians, should use probability to describe uncertainty!
</div>

Architects are not physicists, but they use physics too, and they must use it properly, otherwise 
the buildings they build would fall apart.
Why should data scientists use something different from probability to describe uncertainty?
There is one reasons if statisticians use it, namely that it has many interesting properties
which are well-suited to describe uncertainty.
There is however a question which we must discuss here: what is uncertainty?
As Aki Vehtari explains in [his course](https://www.youtube.com/watch?v=AcKRob0C8EY&list=PLBqnAso5Dy7O0IVoVn2b-WtetXQk5CDk6),
philosophers distinguish between two types of uncertainty:
- **Aleatoric** uncertainty, which is due to an intrinsic randomness of the process under study.
- **Epistemic** uncertainty, which is originated by our ignorance.

In order to quantify the uncertainties, what statisticians do is build a statistical model
of the problem, namely to assume a probability
distribution for the data-generating process $$p(y | \theta)\,.$$
Statistical inference therefore aims to determine the value of the unknown (possibly multidimensional)
parameter $\theta\,.$
This procedure allows us to take into account the aleatoric uncertainty in the
data-generating process, since we assume that the observed data has been generated
according to some probability distribution.
You should notice that we already have a source of epistemic uncertainty, since we do not know which is the true
process which generated the data.
We also have a second source of epistemic uncertainty: even if we knew the true probability
distribution which generated our data, we do not know the value of the parameters, since we are
trying to infer it.

In Bayesian inference, we use probability to describe this kind of uncertainty too, while this is not
true in "classical" statistical inference.



