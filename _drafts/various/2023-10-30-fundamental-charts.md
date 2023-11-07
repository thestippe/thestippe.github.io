---
layout: post
title: "Fundamental charts"
categories: course/various/
tags: /fundamental_charts/
description: "An overview to some of the most common data visualizations"
---

In this post we will take a look at some of the most fundamental charts
that one encounters in data visualization [^1].

[^1]: Here we will follow Enrico Bertini's Coursera lecture on fundamental graphs, which I suggest you to watch together with his entire [specialization in data visualization](https://www.coursera.org/specializations/information-visualization).


## Categorical and quantitative data

The first kind of graph we will discuss is the bar chart, where we show
the how a quantitative attribute changes across a set of categories.

As an example, we can visualize the number of gold medals
that each country won in the 2020 Olympic games.
Here we will only plot a sub-sample of the dataset, while the full dataset
can be found [here](https://github.com/MainakRepositor/Datasets/blob/master/Tokyo-Olympics/Medals.csv).


<!-- Load d3.js -->
<script src="https://d3js.org/d3.v6.js"></script>

<!-- Create a div where the graph will take place -->
<div id="barchart"> </div>

<script src="/docs/assets/javascript/barchart.js"> </script>

In this case the categorical variable is the team, while the quantitative variable
is the number of gold medals.

The bar chart can be rotated by 90 degrees, but the vertical version (which we used)
allows for a larger number of categories to be shown.

If the categories don't have any natural order it may be a good idea to
reorder the categories with respect to the plotted quantity to improve readability.

## Relation between two quantitative variables

The second kind of graph we will talk about is the
line chart, where you can visualize how does a quantitative variable changes with
respect to another quantity.


<div id="linechart"> </div>
<script src="/docs/assets/javascript/linechart.js"> </script>

Line chart is often abused, as the line naturally both encodes order
and a concept of distance between the values in
the x axes, so if x is not a quantitative variable one should
never use the line chart.


## Scatterplot

In a scatterplot we show the distribution of two
quantities across the items.

<div id="my_scatterplot"> </div>

<script src="/docs/assets/javascript/scatterplot.js"> </script>

## Matrix chart

In a matrix we want to visualize how does a quantity
distributes across two categorical variables.

<div id="my_matrix_chart"> </div>
