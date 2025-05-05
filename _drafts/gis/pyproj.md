---
layout: post
title: "101 ways to reproject your data"
categories: /gis/
tags: /geography/
image: "/docs/assets/images/geo/sphere.webp"
description: "The most common ways to define a projection in pyproj"
date: "2024-10-11"
---

In [an old post](/gis/projections) we discussed how to choose a projection.
In this post we will get our hands a little bit more dirty, and we will show
some of the most common ways to select the crs. 
We will [PyPROJ](https://pyproj4.github.io/pyproj/stable/build_crs.html),
which is a Python wrapper for [PROJ](https://it.wikipedia.org/wiki/PROJ).

As datasets, we will the [Biology of birds practical](https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study1349878794),
which is a small dataset describing the GPS track of some bird,
made available by Luke Ozsanlav-Harris for teaching purposes.

```python
import geopandas as gpd
from matplotlib import pyplot as plt
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info, get_authorities

gdf = gpd.read_file('birds/points.shp')
bounds = gpd.read_file('https://raw.githubusercontent.com/martynafford/natural-earth-geojson/refs/heads/master/50m/physical/ne_50m_coastline.json')

bounds.crs
```

<div class="code">
<Geographic 2D CRS: EPSG:4326>
<br>
Name: WGS 84
<br>
Axis Info [ellipsoidal]:
<br>
- Lat[north]: Geodetic latitude (degree)
<br>
- Lon[east]: Geodetic longitude (degree)
<br>
Area of Use:
<br>
- name: World.
<br>
- bounds: (-180.0, -90.0, 180.0, 90.0)
<br>
Datum: World Geodetic System 1984 ensemble
<br>
- Ellipsoid: WGS 84
<br>
- Prime Meridian: Greenwich
</div>

```python
gdf.crs==bounds.crs
```

<div class="code">
True
</div>

The two datasets are encoded with the same CRS, which is the latitude-longitude
one.
Let us first of all take a look at the raw data.

```python
crs = gdf.crs
minx, miny = tuple(gdf.to_crs(crs).bounds[["minx", "miny"]].min().values)
maxx, maxy = tuple(gdf.to_crs(crs).bounds[["maxx", "maxy"]].max().values)
fig, ax = plt.subplots()
bounds.to_crs(crs).plot(ax=ax, color="lightgray")
gdf.to_crs(crs).plot(ax=ax, marker='x')
ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.set_title("Lat-Lon")
fig.tight_layout()
```

![Our map shown in the latitude-longitude coordinates](/docs/assets/images/gis/proj/latlon.webp)

In pyproj there are many ways to choose the CRS, and the most common ones
are ```from_epsg```, ```from_user_input``` and ```from_string```.
These functions are really easy to use,
and ```from_epsg(XXXX)``` is equivalent to ```from_user_input(XXXX)```
and to ```from_string("ESPG:XXXX"")```.

```python
crs = CRS.from_epsg(3395)
# equivalent: CRS.from_user_input(3395) or CRS.from_string("EPSG:3395")
minx, miny = tuple(gdf.to_crs(crs).bounds[["minx", "miny"]].min().values)
maxx, maxy = tuple(gdf.to_crs(crs).bounds[["maxx", "maxy"]].max().values)
fig, ax = plt.subplots()
bounds.to_crs(crs).plot(ax=ax, color="lightgray")
gdf.to_crs(crs).plot(ax=ax, marker='x')
ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.set_title("WGS 84 Mercator")
fig.tight_layout()
```

![The map shown in the WGS-84, crs, also known as
EPSG 3395](/docs/assets/images/gis/proj/wgs84.webp)

Each area has its own typical projections, so if you are used
to map a certain area, you will likely be familiar with
the EPSG codes of these projections.
As an example, I know by heart that the 3003/3004 projection are the Monte Mario
projections, while EPSG: 32632 to EPSG: 32634
are UTM WGS 84 32N-34N, and these are typical projections used in Italy.

In particular, UTM projections are very common when you must work
on the country level. 
If you are working on a map of some country you are not familiar
with, you might want to find the associated UTM projections.

Pyproj allows you to easily find the UTM projections
associated with a given area, and you can do this as follows.

```python
crs = bounds.crs
minx, miny = tuple(gdf.to_crs(crs).bounds[["minx", "miny"]].min().values)
maxx, maxy = tuple(gdf.to_crs(crs).bounds[["maxx", "maxy"]].max().values)
utm_crs_list = query_utm_crs_info(
    datum_name="WGS 84",
    area_of_interest=AreaOfInterest(
        west_lon_degree=minx,
        south_lat_degree=miny,
        east_lon_degree=maxx,
        north_lat_degree=maxy,
    ),
)

utm_crs = CRS.from_epsg(utm_crs_list[len(utm_crs_list)//2].code)

crs = utm_crs
minx, miny = tuple(gdf.to_crs(crs).bounds[["minx", "miny"]].min().values)
maxx, maxy = tuple(gdf.to_crs(crs).bounds[["maxx", "maxy"]].max().values)
fig, ax = plt.subplots()
bounds.to_crs(crs).plot(ax=ax, color="lightgray")
gdf.to_crs(crs).plot(ax=ax, marker='x')
ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.set_title(f"{utm_crs.name}")
fig.tight_layout()
```

![Our map in UTM 26 N WGS 84 projection.](/docs/assets/images/gis/proj/utm.webp)

Not all projections have an associated EPSG code, so you might
be interested in using a projection which is encoded by some authority
which is not EPSG.
You can do this by using the ```from_authority``` function.
The list of the available authorities can be obtained as

```python
get_authorities()
```

<div class="code">
['EPSG', 'ESRI', 'IAU_2015', 'IGNF', 'NKG', 'NRCAN', 'OGC', 'PROJ']
</div>

As an example, the Gall-Peters projection has an ESRI code, but
it has no EPSG code.

```python
crs = CRS.from_authority('ESRI', code=54016)
minx, miny = tuple(gdf.to_crs(crs).bounds[["minx", "miny"]].min().values)
maxx, maxy = tuple(gdf.to_crs(crs).bounds[["maxx", "maxy"]].max().values)
fig, ax = plt.subplots()
bounds.to_crs(crs).plot(ax=ax, color="lightgray")
gdf.to_crs(crs).plot(ax=ax, marker='x')
ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.set_title("Gall-Peters")
fig.tight_layout()
```

![Our map in the Gall-Peters projection](/docs/assets/images/gis/proj/gall_peters.webp)

In some situation, you might be interested in a deeper customization,
because you want to preserve some particular property in your analysis.

In these cases, simply using a code might not be enough, and you might look
for choosing the exact tangent point of your map.
In this case you can use the proj syntax.
In our case, we will use the equidistant conic projection
with latitude equal to the median of the latitude of our GPS
points.

```python
crs = CRS.from_proj4(f"+proj=eqdc +lat_1={gdf.geometry.y.median()}")
minx, miny = tuple(gdf.to_crs(crs).bounds[["minx", "miny"]].min().values)
maxx, maxy = tuple(gdf.to_crs(crs).bounds[["maxx", "maxy"]].max().values)
fig, ax = plt.subplots()
bounds.to_crs(crs).plot(ax=ax, color="lightgray")
gdf.to_crs(crs).plot(ax=ax, marker='x')
ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.set_title("Equidistant conic")
fig.tight_layout()
```

![The map shown in our custom projection](/docs/assets/images/gis/proj/custom.webp)

## Conclusions

We have seen how to choose our own projection with pyproj, starting from
the simpler methods up to the powerful proj syntax.

```python
%load_ext watermark
```

```python
%watermark -n -u -v -iv -w
```

<div class="code">
Last updated: Tue Apr 15 2025
<br>

<br>
Python implementation: CPython
<br>
Python version       : 3.12.8
<br>
IPython version      : 8.31.0
<br>

<br>
matplotlib: 3.10.1
<br>
geopandas : 1.0.1
<br>
pyproj    : 3.7.1
<br>

<br>
Watermark: 2.5.0
</div>