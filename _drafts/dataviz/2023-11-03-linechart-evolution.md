---
layout: post
title: "Evolutions of the line chart"
categories: /dataviz/
tags: /linechart-evolution/
description: "When a linechart is not enough"
---

In a previous post we saw some of the most fundamental charts,
which are the basic building blocks for data visualization.

Datasets can become very complex, and you should adapt your data visualization
depending on your needs.
Here we will take a look at how we can draw more and more complex datasets
by simply changing few details of the basic visualization, and we will
do so by using the line chart as fundamental visualization.

## The line chart

As we have already seen, in the line chart we have
- an ordered variable on the $x$ axis
- a quantitative variable on the $y$ axis

As an example, let us take a look at the Italian GDP per capita expressed in US dollars adjusted by the US inflation, which can be found
[here](https://github.com/thestippe/thestippe.github.io/blob/main/data/gdp_per_capita_filtered.csv).

The dataset is based on [this](https://github.com/RaafatSaleh/GDP-per-capita-and-its-effect-on-the-man-life-quality/blob/master/Data/gdppercapita_us_inflation_adjusted.csv) repo.

<!-- Load d3.js -->
<script src="https://d3js.org/d3.v5.js"></script>

<div id="linechart"> </div>
<script src="/docs/assets/javascript/linechart_evolution/linechart.js"> </script>

This visualization allow us to see how a quantity (the GDP per capita)
changes over time, and it does that in a decent way.

## Issues with the line chart

But what does it happen when we try and visualize more than one Country in
a single plot?
Let us start by using color to code the Country

<div id="multiple_linechart"> </div>
<script src="/docs/assets/javascript/linechart_evolution/multiple_linechart.js"> </script>

As the number of lines grows, the graph soon becomes more and more cluttered.
Already with a small number of lines it becomes difficult to catch the behavior
of a single line.

We have two main alternatives to the multiple line chart:
- we can put one line chart per graph and we create a small multiples
- we can use another channel to encode the temperature
or, of course, we can combine the two techniques.


## Small multiples

Broadly speaking, when you build a small multiple you draw more than
one visualization, and each visualization is indexed by a label which is not
used in any of the single images.

<div id="sm_linechart"> </div>
<script src="/docs/assets/javascript/linechart_evolution/sm_linechart.js"> </script>

Here we used small multiples to put one visualization on the right of the previous,
but you can also order them vertically or build a grid.

The main advantage of the small multiples is that they reduce clutter,
but it becomes more difficult to compare the single lines.


## Stacked area chart

Another possible solution is to stack the lines one above the other one,
and this is done in the stacked area chart.

<div id="stacked_chart"> </div>
<script src="/docs/assets/javascript/linechart_evolution/stacked_chart.js"> </script>

The major issue with this solution is that, for all but the lowest curve,
the baseline is not constant, and this makes difficult to quantify the values.
