---
layout: post
title: "OpenEO for SAR images"
categories: /gis/
tags: /geography/
image: "/docs/assets/images/gis/openeo/landslide_procida.webp"
description: "Accessing Synthetic Aperture Radar data from OpenEO"
date: "2024-12-06"
---

In the [last post](\gis\openeo) we discussed how to use OpenEO
to access remote sensing data using the Sentinel 2 dataset.
In this post we will use the same library to access the Sentinel 1
dataset, which is a Synthetic Aperture Radar mission which collected
data from 2 satellites for many years.
SAR is quite different from visible data, and as the name suggests,
it is collected by a radar. This means that the light source
is on the satellite, so it can collect data regardless on the sun position.
Generally, SAR operates on frequencies much higher than the ones in the visible
spectrum, so they do not interact with clouds, and allows for a great
spatial resolution.
These characteristics make SAR imaging a very interesting tool
to monitor the Earth surface.

As you can imagine, there's no free lunch, and we must pay a price in order
to get the above advantages.
One of the main disadvantages is that SAR images are really noisy,
so you must put a lot of effort in image processing.



## Oil spill detection

We will use the Sentinel-1 dataset to detect an oil spill in the Kwait gulf in 2017.
We will re-perform an analysis given as [tutorial in the Copernicus documentation](https://documentation.dataspace.copernicus.eu/APIs/openEO/openeo-community-examples/python/OilSpill/OilSpillMapping.html),
but we will use xarray rather than Copernicus to perform the last steps of the analysis.

```python
# Load the essentials
import openeo
import openeo.processes
import numpy as np

connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
```

<div class="code">
Authenticated using refresh token.
</div>

```python

```