---
layout: post
title: "Hypercoast"
categories: /gis/
up: /gis
tags: /geography/
image: "/docs/assets/images/gis/openeo_ts/results.webp"
description: "Unleash the power of hyperspectral remote sensing"
date: "2024-12-01"
---

Hyperspectral remote sensing can be considered as an evolution of multispectral
remote sensing, where the wavelength resolution is small enough to allow
approximating the spectrum as a (piecewise) smooth function within a good
approximation.

A remote sensing tool can be considered hyperspectral if the wavelength resolution
is below one tenth of the wavelength itself, and most of the hyperspectral
remote sensing satellites measure the surface reflectance over hundreds
of different wavelengths.

It is important to stress that the spectrum is depicted as a **piecewise**
smooth function because there are two components which break the smoothness:
first of all, there are components of the spectrum where
the atmosphere is opaque, so there is no physical way to investigate
those wavelengths by means of a satellite.
Secondly, most of the atomic and molecular spectra both have a continuous
component and a discrete one, and with a fine enough resolution
we could be able to observe both of them [^1]

HRS is a great tool, but it has many drawbacks. HR sensors are
more expensive than their multispectral counterparts, so there are less
HR missions, and consequently a lower spatial and temporal coverage.
Moreover, many HR sensors, especially the ones which are publicly available,
are focused on geophysical or climatological aspects, and therefore their
spatial resolution is iller than the ones of tools such as Sentinel-2.

On the other hand, HRS allows to analyze and monitor 
at a chemical level the Earth ground as well as the coastal seabed, 
and this opens many possible applications.

Now that we discussed the main aspects of HRS, let us see how to use Python
to obtain and analyze hyperspectral data.



[^1]: The physical reason for this requires a quantum mechanic course, and this goes far beyond the scope of this blog.