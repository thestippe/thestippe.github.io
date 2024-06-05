---
layout: post
title: "What is statistics"
categories: /statistics/
subcategory: Introduction
tags: /stats_intro/
date: "2024-01-05"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Let us clarify what we are talking about"
section: 0
published: false
---

In this post I will introduce what is statistics and what are
the where it applies.
This post is the first one of a series of posts about statistical methods.

## Definition of statistics

Statistics is a huge subject, as entire bachelor and master degrees are 
devoted to this topic.
According to [Britannica](https://www.britannica.com/science/statistics)
<q>statistics is the science of collecting, analyzing, presenting, and interpreting data.
</q>
The [Wikipedia definition](https://en.wikipedia.org/wiki/Statistics) is very similar, as it defines statistics as
<q>the discipline that concerns the collection, organization, analysis,
interpretation, and presentation of data.</q>

Here we will mostly focus on analysis and interpretation. We will discuss from time to time some issues related
to data collection, as the choices you made when collecting the data strongly affects the analysis and the interpretation
of the results.
I will try and keep issues related to data presentation into the [dataviz](/dataviz) section, especially those
related to visually presenting raw data and more general questions related to visualization,
while here we will discuss how to present the analysis.

In statistics you use data to make statements about a well defined collection of units, namely your
[**population**](https://en.wikipedia.org/wiki/Statistical_population).
[Descriptive statistics](https://en.wikipedia.org/wiki/Descriptive_statistics)
is the branch of statistics which discusses how to state facts and proven outcomes from a population.
We don't always have data regarding the entire population, but we often end up with informations
about a subset of our population, namely a [**sample**](https://en.wikipedia.org/wiki/Sampling_(statistics)).
The discipline which discusses how to draw conclusions about the population by only having a sample
is named [**inferential statistics**](https://en.wikipedia.org/wiki/Statistical_inference), and will be the main
topic of this section of the blog.

## Inferential statistics

As we previously anticipated, in inferential statistics you want to draw conclusions about your population
by analyzing a sample.
In order to draw conclusions about the entire population from the sample we must make some assumption about
how our population is composed and how does our sample relates to our population,
and these assumptions are mathematically formulated into a [**statistical model**](https://en.wikipedia.org/wiki/Statistical_model).

In other words, a statistical model is a mathematically rigorous idealization of the characteristics of the population:
we are trying to catch the relevant features of some phenomenon, but it is almost always impossible
to catch *all* the features that contribute to the phenomenon:

<div class='emphbox'>
<a href="https://en.wikipedia.org/wiki/All_models_are_wrong">
All models are wrong, but some are useful.
</a>
</div>

## Statistical models

As previously stated, when we build a statistical model, our aim is to get information
about the entire population by only studying a sample.
Of course, we don't know all the characteristics of our population,
otherwise we wouldn't need to study it!

<div class="emphbox">
Estimating and communicating uncertainties is the core of inference.
</div>

This implies that we somehow must account for our ignorance, and this can be done by 
assuming that the relevant features of our population are **randomly distributed
according to some probability distribution**, so our task is now to somehow constrain
the space of the reasonable distributions that describe our population.

At this point, if you are not familiar with these concept, you may wonder what does probability **mean**,
but this is far from being an easy question to answer,
as randomness can come into many different forms.

The measure of the spin of an electron is random, as quantum mechanics tells us that is intrinsically
impossible to know the outcome of the experiment with.

Also tossing a coin is a random event, although this could be in principle determined by Newton's laws.
In fact, even a small variation of the initial position of the coin or of its initial speed can dramatically change
its trajectory.

Tomorrow's value of Microsoft's financial stock is a random quantity too,
but this kind of ignorance is subjective, as we might know some relevant information
that we might think will affect the price.

These three examples describe three very different situations, so you shouldn't be surprised that
there is more than one answer to the question "what probability *means*".

However, there is much more agreement on the answer to the question "what probability *is*",
and this is given by the branch of mathematics named [**probability theory**](https://en.wikipedia.org/wiki/Probability_theory).

## Conclusions

We discussed what is statistics and what do we mean by statistical
model.
In the future posts we will discuss more in depth about probability
before entering into statistical methods.
