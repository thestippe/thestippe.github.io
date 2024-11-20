---
layout: post
title: "Geostatistical modelling"
categories: /statistics/
subcategory: "Geostatistics"
tags: /mrp/
date: "2024-09-08"
section: 9
# image: "/docs/assets/images/perception/eye.jpg"
description: "Including spatial dependence into the models"
published: false
---

In many circumstances, you may want (or need) to take into account
some spatial aspect of your problem. You might need to estimate the incidence
of a disease, or maybe you want to estimate the temperature value at some location 
based on the records of different thermometer spread across some region.
In these cases, you should keep in mind 
[Tobler's first law of geography](https://en.wikipedia.org/wiki/Tobler%27s_first_law_of_geography):

<br>
> Everything is related to everything else, but near things are more related than distant things.
<br>

If it's raining over me, it is likely that at a distance of 10 meters it is
raining too, but it is hard to guess whether it is raining at 1000 km from me or not.
When building a statistical model, you might be forced to include this
observation into your model, and geostatistical modelling allows you to do so.

## Kinds of spatial data
In geostatistics it is customary to distinguish between tree major classes
of data:
- **Areal data**: space is divided in a finite number of disjoint regions, and for each region we have one (or more) datum, like the number of inhabitants of a city or the GDP of a country.
- **Geostatistical data** refers to data which are sampled at specific locations, like temperature measurements or the prices of apartments.
- **Point patterns** are location of random events, such as thunders or earthquakes, but also disease outbreaks.

The main difference between geostatistical data and point pattern, is that
in geostatistical data, the location is an independent explanatory variable, while
in point pattern it is the response variable.
Notice that, for point patterns, some variable might be attached to
the random point.
As an example, earthquakes have random spatio-temporal distribution,
so modelling the spatial distribution of earthquakes happened in a certain
time interval would be a point pattern problem. 
Earthquakes, however, also have a magnitude, so you both have a random
geolocation and an associated numeric variable.

Sometimes areal data are also called regional data or lattice data, and
the last name is particularly appropriate when the space is divided into
a regular grid rather than on the basis of some social convention such as 
country boundaries.

For each kind of data there are different models, and in the following we will discuss the most common model for
each kind of data.
