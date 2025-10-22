---
categories: /statistics/
up: /statistics
date: 2026-03-30
description: How to account for multiple effects
layout: post
section: 10
subcategory: Other random thoughts
tags: /decomposition/
title: Intelligent decomposition
---

One of the great advantages of using probabilistic programming languages is that you can easily encode
knowledge inside your models, and this is what I would like to discuss in the present post.
We already saw how to use spline in [the dedicated post in this blog](/statistics/spline),
here we will show one possible way to use them to account for multiple effects.
I am a big fan of gaussian processes, since they easily allow you to
account for multiple effects such as short range correlations, long range correlations or periodicity.
When the dataset becomes too large, however, even HSGP models suffers, and the computational
time becomes too large to be useful.
In this case, an intelligent usage of splines is a cheap and practical alternative.

## The dataset
At [this link](https://idrometri.agenziapo.it/Aegis/data/data?elementId=35552)
you can find and download the historical records of the Po level for the Turin city center,
and we will use the dataset to provide an estimate the flood hazard.
In order to do so, we will use [extreme value theory](/statistics/extreme_intro),
which we discussed at the linked page.
