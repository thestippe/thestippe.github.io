---
layout: post
title: "Evolutions of the line chart"
categories: /dataviz/
tags: /linechart-evolution/
image: "/docs/assets/images/linechart_evolutions/areaplot.png"
description: "When a linechart is not enough"
---

In a [previous post](/fundamental-charts) we saw some of the most fundamental charts,
which are the basic building blocks for data visualization.

Datasets can become very complex, and you should adapt your data visualization
depending on your needs.
Here we will take a look at how we can draw more and more complex datasets
by simply changing few details of the basic visualization, and we will
do so by using the line chart as fundamental visualization.

## The line chart

As we have already seen, in the line chart we have
- an ordered key attribute on the $x$ axis
- a quantitative value attribute on the $y$ axis

As an example, let us take a look at evolution over the years the Italian GDP per capita expressed in US dollars adjusted by the US inflation, which can be found
[on the github repo of the blog](https://github.com/thestippe/thestippe.github.io/blob/main/data/gdp_per_capita_filtered.csv).

The dataset is based on [Raafat Saleh's repo](https://github.com/RaafatSaleh/GDP-per-capita-and-its-effect-on-the-man-life-quality/blob/master/Data/gdppercapita_us_inflation_adjusted.csv).


<div id="linechart"> </div>
<script src="/docs/assets/javascript/linechart_evolution/linechart.js"> </script>

This visualization allow us to see how a quantity (the GDP per capita)
changes over time, and it does that in a decent way.

## Issues with the line chart

But what does it happen when we add a second categorical key attribute?
As an example, let's try and visualize more than one Country in
a single plot.
Let us start by using color to encode the Country

<div id="multiple_linechart"> </div>
<script src="/docs/assets/javascript/linechart_evolution/multiple_linechart.js"> </script>

As the number of lines grows, the graph soon becomes more and more cluttered.
Already with a small number of lines it becomes difficult to catch the behavior
of a single line.

We have two main alternatives to the multiple line chart:
- we can put one line chart per graph and we create a **small multiples**
- we can use another channel to encode the value attribute
or, of course, we can combine the two techniques.


## Small multiples

Broadly speaking, when you build a small multiple you draw more than
one visualization, and each visualization is indexed by a label which is not
used in any of the single images.
This technique is also called faceting.

<div id="sm_linechart"> </div>
<script src="/docs/assets/javascript/linechart_evolution/sm_linechart.js"> </script>

Here we used small multiples to put one visualization on the right of the previous,
but you can also order them vertically or build a grid.

The main advantage of the small multiples is that they reduce clutter,
but it becomes more difficult to compare the single lines.


## Stacked area chart

If the value attribute is sequential as in our case (the GDP cannot become negative), another possible solution is to stack the lines one above the other one,
and this is done in the stacked area chart.

<div id="stacked_chart"> </div>
<script src="/docs/assets/javascript/linechart_evolution/stacked_chart.js"> </script>

The major issue with this solution is that, for all but the lowest curve,
the baseline is not constant, and this makes difficult to quantify the values.

## Streamgraph

A stacked bar chart can become cumbersome when one has many channels, and in this
case one may use a streamgraph.

<div id="steamgraph"> </div>
<script src="/docs/assets/javascript/linechart_evolution/steamgraph.js"> </script>

The streamgraph is obtained by allowing the lower
line to vary, and either by making it symmetric with respect to the $x$ axis or by choosing it exact shape by minimizing some target quantity.
This method allows you to show
a large number of categories, but the main drawback
is that one needs some practice to read it.

## Conclusions

We have seen few possible evolutions of the line chart.
Those alternatives are appropriate when you want to plot the evolution of a quantitative variable
for a set of categories.
Faceting can be combined with anyone of the visualizations we have previous discussed, while stacking can 
only be applied to bar chart or line chart.
Finally, we have seen the streamgraph,
which uses an alternative way of stacking the lines.
