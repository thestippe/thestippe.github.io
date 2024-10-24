---
layout: post
title: "Drawing geographic maps"
categories: /dataviz/
tags: /geography/
image: "/docs/assets/images/geo/geo.webp"
description: "Dealing with geographic data"
---

As defined by Enrico Bertini, geographical maps are visualizations
where non-geographical objects are visualized, and some geographical entities
are used to provide some spatial reference to the reader.
Here we define a geographical entity as any kind of entity which encodes
geographical information.
These are:
- **Points** such as latitude and longitude, and give information about the location of a single entity.
- **Lines** as might be the location of a street or of a river.
- **Areas** which gives the location of a geographically extended object, such as a state or of a building.

Associating to a non-geographical entity a geographical property
is named **geo-encoding**, while the extraction of some non-geographical
information given the geographical location is said **geo-decoding**.

There is quite a lot of agreement that, unless strictly needed, geographical
maps are not effective for data visualization.
The reason for this is quite easy to guess, and it's simply because length
can hardly be used, since the spatial channels are already encoding
the geographical information.
There are, however, cases where one simply cannot avoid using geographical maps.

## Latitude and longitude
Let us now recall how latitude and longitude,
which are the most common coordinates on the spherical Earth, are defined.
This will be easier if we first define the **great circle**,
which is any circle given by the intersection
of the sphere (the Earth) with a plane going through the center of the sphere.

Great circles are very useful, as the shortest path between two points on
the sphere is an arc of great circle[^1].

[^1]: This is why, for long distances, it is convenient to work with nautical miles, which are defined as the length of one prime of degree of the great circle.

We must then define a **pole**, as an example the North Pole (the South Pole can be 
defined as the opposite point on the sphere).
We then define the **equator** as the great circle with the greatest distance
from the poles or, equivalently, as the great circle defined by the plane 
perpendicular to the line passing through the poles.

The **north/south hemisphere** is the surface of the half sphere which has as boundary the equator
and encloses the North/South Pole

A **parallel** is any circle on the sphere given by the intersection
of the sphere with a plane perpendicular to the line passing through the poles.
The equator is the only parallel which is also a great circle.

We define **meridians** as the great circles passing through both the poles.

![](/docs/assets/images/geo/sphere.webp)

In the above figure we show the meridians and parallels on the sphere, in grey.
All meridians encounter at the poles, parallels never encounters.
The red dashed line correspond to the equator, the purple dashed line to the Greenwich meridian.
The dotted red line is the parallel passing through the black point,
the dotted purple line corresponds to the meridian passing through it.
The red solid arc corresponds to the point longitude,
the purple solid arc corresponds to its latitude.

We must now identify a second great circle passing through both the poles,
and this is conventionally defined as the one going through Greenwich.

We can now define the **latitude** $\phi$ of a point as the shortest arc of meridian 
going from the equator and the parallel passing through the point.
If a point is in the north/south hemisphere, then its latitude is positive/negative (N/S).

The **longitude** $\lambda$ is the shortest arc of equator going from the Greenwich meridian
and the meridian passing through the point.
The longitude is positive (East) if, by looking from the North Pole,
the shortest arc starting from Greenwich rotates anticlockwise,
otherwise it's negative (West).


## Maps and projections
Earth is (almost) spherical, but our screen is flat, so we need to use a
**projection** to map the points on the sphere onto the plane.

The simplest kind of projection we will discuss is the **perspective projection**,
which is obtained by projecting the rays originated by 
a dummy light source going through the sphere surface onto
a plane.

In order to define a projection, we must define
- a projection point, which is the light source
- the surface where we will project the sphere

Common projection points are:
- the center of the sphere (central)
- a point onto the sphere - usually a pole
- a point located at infinite distance from the sphere and the rays are parallel lines (orthographic)

![](/docs/assets/images/geo/projection_points.webp)

We must also choose the plane where we want to project the sphere.
Common choices are
- a plane tangent to a point, usually the pole (planar)
- a cone tangent to any given circle, usually parallel (conic)
- a cylinder tangent to any great circle, usually the equator (cylindrical)

Planar and cylindrical projections can be considered special cases of
the conical one.

In any polar projection, parallels are mapped onto circles,
while meridians are straight lines starting from the pole.

Some common projections are:
- the **gnomonic**, the planar perspective projection where the rays start from the center of the sphere
- the **stereographic** projection, the planar perspective projection where the rays start from the point opposite to the point of the sphere tangent to the projection plane
- the **orthographic** projection, the planar perspective projection where the rays are parallel
- the **central cylindrical** projection

Here we show how equidistant parallels and meridians are mapped for the above projections


![](/docs/assets/images/geo/polar_planar_projections.webp)

Each projection has pros and cons, as usual.
As an example, in the gnomonic projection, arcs of great circles are always
mapped onto straight lines, making them convenient to trace routes. However,
in this projection, distances and areas are distorted,
and the more one goes far from the center of the projection
(where we assume the plane touches the sphere), the worst 
becomes the distortion.

On the other hand, the central cylindrical projection is well suited to show
regions close to the equator, but the poles are mapped at the infinity.

Finally, the stereographic projection is a **conformal** projection,
and this means that angles are preserved in the mapping, and this makes them
suitable to calculate angle differences.

Conformal projections form a very important class of projections, and the stereographic
projection is only one of these projections.
Other well known examples of conformal projections are the
**Mercator** projection, which is a conformal projection derived from
the cylindrical projection, and the **Lambert conical** projection, which is instead derived
from the central conical projection.

![](/docs/assets/images/geo/south_pole_conformal.webp)

In the above figure we show how the South Pole, in blue, is mapped for different
conformal maps.

Another very important class of projection is the family of **equal-area**
projections. In equal-area projections, the ratio between areas is preserved.

![](/docs/assets/images/geo/south_pole_equal_area.webp)

Due to this property, equal area projections are often used in statistics,
where one wants to compare quantities for different countries and relate
these quantities to the extension of the country itself.
Above, we show some example of equal-area projection, namely the **Lambert conical**,
the **Equal Earth** and the **Hammer** projections.

## Conclusions

Choosing the appropriate projection is crucial to convey the most relevant information,
especially when you need to visualize large geographical areas.
There exists no perfect projection, and each projection has pros and cons,
it is therefore important both to have some basic repertoire of projections
and to experiment with different projections.
As usual, drawing is re-drawing.