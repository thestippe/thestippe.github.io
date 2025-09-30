---
layout: post
title: "Beyond the 1D scatterplot"
categories: /dataviz/
up: /dataviz
tags: /beyond-scatterplot/
image: "/docs/assets/images/beyond_scatterplot/boxplot.svg"
description: "How to visualize one dimensional distributions"
---


When we discussed [the fundamental charts](/dataviz/fundamental-charts), we saw the most basic way to visualize one dimensional
quantitative data.
In this post we will discuss alternatives ways to perform this task. 

<script src="https://d3js.org/d3.v5.js"></script>

## 1-D Scatterplot

As we already discussed, the one dimensional scatterplot is the easiest
way to visualize one dimensional quantitative data.

<div id="my_scatterplot1d"> </div>

<script src="/docs//assets/javascript/fundamental_charts/scatterplot1d.js"> </script>

This visualization can however become very crowded as the number of points grows,
and in this case it is hard to count the number of points within a certain range
of values.
This is known as the **curse of dimensionality**,
and it's one of the most central problems
in dataviz.

In order to allow for a larger number
of points, you can jitter your points,
namely put them randomly on your
y axis. This operation may
however confuse your audience
if they are not used to it.

## Histogram

Histograms are very common ways to show how a certain quantity is distributed.
In a first step, a range of values is divided in
$n$ equally spaced intervals.
We then visualize the number of objects belonging to
each interval.

<script src="/docs/assets/javascript/beyond_scatterplot/histogram.js"> </script>

<div id="hist"> 
</div>

While this visualization scales much better than
the scatterplot, it has as main drawback the fact
that the results (and so the conclusions) depend
on the choice of the interval, and the discretization
may hide some important feature of the data.

## Boxplot

Boxplot is another popular way to show one dimensional
distributions.
In this visualization, rather than showing the data,
we show the main features of the underlying 
distribution.

<div id='boxplotdiv'></div>
<script src="/docs/assets/javascript/beyond_scatterplot/boxplot.js"></script>

In order to understand this visualization, your audience
should be trained, so it's not suited for general audience.
There is no agreement on the exact values one should visualize.
As an example, we used the 10th, 25th, 50th, 75th and 90th percentile
for the five values shown (the leftmost point, the left border of the
box, the central value, the right border of the box and the rightmost point respectively).
Moreover, this visualization is only meaningful for unimodal data,
so you should always make sure that these condition hold.


## Kernel Density Estimate

Kernel Density Estimates, KDE for brevity,
are a set of methods to estimate the probability
distribution function of a set of data.

<div id="kde1d"> 
</div>

<script src="/docs/assets/javascript/beyond_scatterplot/kde.js"> </script>

The result strongly depends on the kernel choice
as well as on its free parameters,
which are both subjective choices.
Before using this method you should make sure
that your kernel is appropriate for your data.
Due to the strong dependence of the result on
the kernel choice, many authors both show
 the KDE plot and the scatterplot.

In case you want to know more about KDE,
there are many resources online, as they have been
very popular in the Machine Learning community.
You can take a look at the [Wikipedia page](https://en.wikipedia.org/wiki/Kernel_density_estimation) 
or at the [scikit-learn documentation](https://scikit-learn.org/stable/modules/density.html).

## Conclusions

We have seen few methods to visualize the distribution
of a single quantitative variable.
If your dataset is small, then the best way to
show your data is simply to show them on a scatterplot,
as this is the most transparent way to visualize
a single quantitative variable.

If your dataset is large, however, choosing how to visualize
your data is a trade off between transparency
and readability, and there is no single answer to the
question of how to choose the most appropriate visualization.

