---
layout: post
title: "Fundamental charts"
categories: /dataviz/
tags: /fundamental_charts/
image: "/docs/assets/images/charts/charttypes.png"
description: "An overview to some of the most common data visualizations"
---


<!-- Load d3.js -->
<script src="https://d3js.org/d3.v5.js"></script>

In this post we will take a look at some of the most fundamental charts
that one encounters in data visualization [^1] [^2].


[^1]: Here we will follow Enrico Bertini's Coursera lecture on fundamental graphs, which I suggest you to watch together with his entire [specialization in data visualization](https://www.coursera.org/specializations/information-visualization).

[^2]: Well, to be honest this post is also an excuse to try and see how to draw charts with [D3.js](https://d3js.org/), and up to the moment it looks an amazing tool!

## 1-D Scatterplot

Although not very common in explanatory visualization, one may decide to simply visualize
a single value attribute across some items, and in this case we 
can use a one dimensional scatterplot.
In this example we will show the distribution of the sepal width for the well known Iris dataset.

<div id="my_scatterplot1d"> </div>

<script src="/docs//assets/javascript/fundamental_charts/scatterplot1d.js"> </script>

The one dimensional scatterplot may be used to visualize the distribution of our attribute
across our items or to find possible outliers.

## 2-D Scatterplot

In a two dimensional scatterplot we show the distribution of two
quantities across the items.
In this case we have no key attribute.
As an example, here we show how the sepal length and the sepal width are
varying across the items of the already used Iris dataset.

<div id="my_scatterplot"> </div>

<script src="/docs//assets/javascript/fundamental_charts/scatterplot.js"> </script>

This visualization can be helpful to determine the underlying distribution
for our attributes, to find whether there exist some correlation among the two
variables or to look for clusters.

## Bar chart

In a bar chart we show
the how a quantitative attribute changes across a set of categories,
which represent our key attribute.

Here and in the future will work under the hypothesis that there are no duplicates among the categories.
In the database language, we may say that our key is a primary key.

As an example, we can visualize the number of gold medals
that each country won in the 2020 Olympic games.
Here we will only plot a sub-sample of the dataset, while the full dataset
can be found [here](https://github.com/MainakRepositor/Datasets/blob/master/Tokyo-Olympics/Medals.csv).



<!-- Create a div where the graph will take place -->
<div id="barchart"> </div>

<script src="/docs//assets/javascript/fundamental_charts/barchart.js"> </script>

In this case the categorical variable is the team, while the quantitative variable
is the number of gold medals.

The bar chart can be rotated by 90 degrees, but the vertical version (which we used)
allows for a larger number of categories to be shown.

If the categories don't have any natural order it may be a good idea to
reorder the categories with respect to the plotted quantity to improve readability.

A bar chart can be very useful when one wants to compare the values of
the attributes across the categories.

## Line chart

In a line chart you can visualize how does a quantitative variable,
which represent our value attribute, changes with
respect to another quantity, which is a key attribute, and it often represents time.
To better explain this graph, let us take a look at the gold
price in the period 1978-2021.


<div id="linechart"> </div>
<script src="/docs//assets/javascript/fundamental_charts/linechart.js"> </script>

This visualization can be useful to extract information between the
value attribute and the key attribute.

Line chart is often abused, as the line naturally both encodes order
and a concept of distance between the values in
the x axes, so if x is not a quantitative variable one should
never use the line chart.



## Matrix chart

In a matrix we want to visualize how does a quantity (our value)
distributes across two categorical variables, which are our key attributes.
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

Matrix charts are typically used to find outliers or clusters.

## Symbol map
The fifth and last type of visualization we will discuss here is the
symbol map, where we show how a quantity varies across two spatial
coordinates.

As an example, here I plot some of the places where I lived, where the area is proportional to the
time I lived in each location.

<script src="https://d3js.org/d3-geo-projection.v2.min.js"></script>

<div id="my_symbol_chart"></div>

<script src="/docs/assets/javascript/fundamental_charts/symbol.js"> </script>

Symbol maps can be used to determine the spatial distribution of a certain quantities.

