---
layout: post
title: "Introduction to geographic data analysis"
categories: /gis/
tags: /gis_intro/
image: "/docs/assets/images/geo/geo.webp"
description: "What will we talk about"
date: "2024-10-03"
---

There are many circumstances where the space plays a major role
in the phenomenon under study, and in these cases it is necessary
to take into account spatial relationships in our description,
**i.e.** to include geographical information into our analysis.
Geographic data science can be considered as an interdisciplinary topic,
embracing mathematics, computer science, statistics, geography, dataviz and many more
disciplines.

Here we will try and give an introduction to this topic.
We will mostly do so by using Python, but I do not exclude we will also use QGIS.
<!--
Tobler's first law of geograpy?
-->

## Layers
When working with geographic data, you often need to draw a **map**,
and a map is generally composed by many **layers**.
You can think about a layer as a specific kind of data you will work
with.
You might have one layer for the elevation, another for the terrain
type, a third layer for the rivers and lakes and other layers
for streets and buildings and so on.

There are two main kinds of layers you will work with: **vector** layers and
**raster** layers.

In GIS, you usually work with coordinates, but coordinates themselves
are meaningless unless you specify the coordinate system associated
to the coordinates.
For this reason, GIS data is **geo-referenced**: it is always associated with the
reference system.

## Vector data

In vector data, each item is associated with a **geometry**,
which is a mathematical entity used to locate the item into the space.

There are three kinds of geometrical entities considered here
- The **point**, which is the fundamental geometrical entity, and it is used to describe the position of object with no spatial extension.
- The **line**, which is a collection of points, and it's used to describe the geometry of one-dimensional objects.
- The **polygon**, which is an area enclosed by a closed line, and it is used to described space regions.

| Point | Line                                                                                                                             | Polygon                                                                                                  |
|-------|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
|<svg height="100" width="100" ><circle r="5" cx="50" cy="50" fill="grey" /></svg> | <svg height="100" width="100"><polyline points="20,10 70,30 20, 40 80, 90" style="fill:none;stroke:grey;stroke-width:2" /></svg> | <svg height="100" width="100"><polygon points="10,10 80,70 10,80" style="fill:grey;stroke:none" /></svg> |


The choice of the appropriate entity depends on your **spatial resolution**:
if you want to describe the position of one house in a map of the World,
then probably a point is the best choice, but this might not be true
if the subject of your map is your city, as in this case the extension
of the house might become relevant, and a polygon might be more appropriate.
Analogously, a river might be represented as a polyline is some
circumstances and as a polygon in others.

## Raster data

Raster data are matrices where each entry, named **cell** or **pixel**,
represents a spatial region, and to each entry is associated one (or more) numerical value.

Photos are stored in raster data, and each cell encodes three colours (red, green and blue).


|   |   |   |   |   |   |
|---|---|---|---|---|---|
| 1 | 2 | 1 | 4 | 0 | 8 |
| 3 | 1 | 3 | 6 | 5 | 7 |
| 5 | 2 | 5 | 3 | 5 | 2 |
| 2 | 4 | 1 | 3 | 1 | 1 |

Raster data are commonly used to represent **fields** such as the
elevation of the field over the sea level or the behavior of the atmospheric
pressure across a region.

## Resolution

Keep in mind that both raster data and vector data are always associated
with a **spatial resolution**.
For vector data, the spatial resolution is given by the numerical
approximation used in the geometry, while for raster data it is
associated to the finite size of the pixel.

We might of course also have other kinds of resolution, such as
the one originated by the resolution of the measurement associated to each pixel
in the raster data.

## Conclusions

We discussed what Geographic Data Science is about, and what characterizes geo-referenced data.
In the next posts, we will discuss how to handle these kinds of data in Python.

<!--
TODO:
- segmentation (knn, yolo? DPM)
- hyperspectral imaging
- 3d imaging
- foluim/other interactive
- postgis? duckdb?
-->