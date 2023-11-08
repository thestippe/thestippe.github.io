---
layout: post
title: "Fundamental charts"
categories: course/various/
tags: /fundamental_charts/
description: "An overview to some of the most common data visualizations"
---

In this post we will take a look at some of the most fundamental charts
that one encounters in data visualization [^1] [^2].

[^1]: Here we will follow Enrico Bertini's Coursera lecture on fundamental graphs, which I suggest you to watch together with his entire [specialization in data visualization](https://www.coursera.org/specializations/information-visualization).

[^2]: Well, to be honest this post is also an excuse to try and see how to draw charts with [D3.js](https://d3js.org/), and up to the moment it looks an amazing tool!


## Categorical and quantitative data

The first kind of graph we will discuss is the bar chart, where we show
the how a quantitative attribute changes across a set of categories.

As an example, we can visualize the number of gold medals
that each country won in the 2020 Olympic games.
Here we will only plot a sub-sample of the dataset, while the full dataset
can be found [here](https://github.com/MainakRepositor/Datasets/blob/master/Tokyo-Olympics/Medals.csv).


<!-- Load d3.js -->
<script src="https://d3js.org/d3.v5.js"></script>

<!-- Create a div where the graph will take place -->
<div id="barchart"> </div>

<script src="/docs//assets/javascript/fundamental_charts/barchart.js"> </script>

In this case the categorical variable is the team, while the quantitative variable
is the number of gold medals.

The bar chart can be rotated by 90 degrees, but the vertical version (which we used)
allows for a larger number of categories to be shown.

If the categories don't have any natural order it may be a good idea to
reorder the categories with respect to the plotted quantity to improve readability.

## Relation between two quantitative variables

The second kind of graph we will talk about is the
line chart, where you can visualize how does a quantitative variable changes with
respect to another quantity, which often represents time.
To better explain this graph, let us take a look at the gold
price in the period 1978-2021.


<div id="linechart"> </div>
<script src="/docs//assets/javascript/fundamental_charts/linechart.js"> </script>

Line chart is often abused, as the line naturally both encodes order
and a concept of distance between the values in
the x axes, so if x is not a quantitative variable one should
never use the line chart.


## Scatterplot

In a scatterplot we show the distribution of two
quantities across the items.
As an example, here we show how the sepal length and the sepal width are
varying across the items of the well known Iris dataset.

<div id="my_scatterplot"> </div>

<script src="/docs//assets/javascript/fundamental_charts/scatterplot.js"> </script>

## Matrix chart

In a matrix we want to visualize how does a quantity
distributes across two categorical variables.
As an example, here we visualize how many points each team of the Six Nations Championship
performed against each opponent in the period 2016-2023.

<div id="my_matrix_chart"> </div>

<script src="/docs//assets/javascript/fundamental_charts/matrix.js"> </script>

As we will discuss in a future post, this representation is never optimal,
as the two spatial dimensions are already encoding the categorical
variables, so one must rely on another channel, typically area or color,
to encode the quantitative variable. 
The issue is that our perception of scale variations in both channels
are prone to errors, so one may find difficulties to correctly
decode the quantitative informations.

## Symbol map
The fifth and last type of visualization we will discuss here is the
symbol map, where we show how a quantity varies across two spatial
coordinates.

As an example, here I plot some of the places where I lived, where the area is proportional to the
time I lived in each location.

<script src="https://d3js.org/d3-geo-projection.v2.min.js"></script>

<div id="my_symbol_chart"> </div>

<script src="/docs//assets/javascript/fundamental_charts/symbol.js"> </script>


