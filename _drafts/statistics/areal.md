---
layout: post
title: "Areal data"
categories: /statistics/
subcategory: "Geostatistics"
tags: /mrp/
date: "2024-09-15"
section: 9
# image: "/docs/assets/images/perception/eye.jpg"
description: "Treating space as a finite set of regions"
published: false
---

As previously anticipated, 
the simpler class of model for spatial data is areal data,
that is data where each raw is associated with one of a finite number of possible
disjoint regions of the space.
Assuming that we can model the observations as normal variables,
we can model them as

$$
y_{i} \sim \mathcal{N}(\mu_i, \sigma)
$$

where the index $i$ runs over the regions.

$$
\mu_i = X_i \cdot \beta + \phi_i
$$

where $X_i$ is the vector of explanatory variables, $\beta$ encodes
the effects, while $\phi_i$ is a latent variable introduced
to model the spatial dependence.
Many different ways have been proposed in the literature to
model the spatial dependence.
If there is no reason to assume that spatial correlation plays a relevant
role, a common choice is to assume that the $\phi_i$ are i.i.d.,
and our model would simply reduce to a hierarchical model, which we already
discussed in a previous post.

Spatial independence is however not always a good choice, since in many cases
we must also encode Tobler's law in our model.