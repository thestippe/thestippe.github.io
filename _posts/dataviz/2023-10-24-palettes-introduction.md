---
layout: post
title: "How to choose a color map"
categories: /dataviz/
up: /dataviz
tags: /color-map/
description: "Each task has its own palette"
image: "/docs/assets/images/palettes_introduction/palettes.png"
---

In the [last post](/color-introduction/) we defined the LCh color space and 
explained why this is, at the moment, the best color space you can use to construct
a color map.
In this post we will explain how to build a color map with it.

<script src="https://d3js.org/d3.v5.js"></script>


## Categorical color maps

The easiest way to construct a categorical color map is to keep
the chroma and the luminance constant, and to vary the hue.

Varying the hue has the advantage that your audience will be able to
refer to a particular category by simply using the color name.
This is of course true unless you have too many colours, in this case naming the color
could be ambiguous.
Moreover, by keeping the luminance constant, there will be no category
which stands out more than the other categories, so your classes
will be all be perceived in the same way.

Naively, you may be tempted to choose the highest luminance
as well as the highest chroma 
as possible in order to accommodate the highest number of classes as possible.
From a perceptual point of view, however, this is not a good idea:
your audience will be likely distracted by the bright colors, so using
less brilliant colors will reduce the cognitive load of your audience.
In order to do this, you can either decrease the chroma or the luminance
of your color, or both.

Keeping a high chroma and a high luminance has another issue, related to the structure of the LCh color space:
given a chroma and a luminance, not all the hues are displayable,
and the more you increase the chroma, the more the available hue range becomes smaller.

Keeping the luminance constant can moreover be problematic for people affected by a quite common inability,
namely color blindness. Some people have a reduced or either completely absent
capacity to perceive color hue, so keeping the luminance constant will
imply that this part of your audience will experience problems into a correct 
interpretation of your visualization.

There are also design issues to take into account:
spanning the entire hue circle will likely make your visualization look
like an harlequin, and this will also contribute distracting your audience.
A possible strategy to avoid this could be to restrict your hue variation
within warm colors or cold colors.
Additionally, you may need to consider that color may convey an implicit message,
and this message may depend on the cultural background of your audience.
This may be a positive fact or a negative one for you: red and blue meanings are universally
recognized when talking about temperature, so you may consider using
red to refer to a high volatility (which is often referred to high temperature
in statistics) period in a market analysis and use blue to depict low volatility periods.
On the other hand, in finance, red is associated with loss, so by using the above
convention you may implicitly transmit the idea that high volatility
implies loss, which is simply false (you may have a period where an
index is always negative, and this is low volatility loss).

There may be finally some task-specific issue to take into account.
Companies often require that their visualizations fulfill specific
requirements, like using the company's palette.

<div class="emphbox">
When you choose a color map, you must consider many factors, and finding 
a good balance can be a tricky task.
</div>

In order to reduce some of these issue, one should keep in mind that color
is an appropriate channel for categorical attributes for a limited number
of classes, possibly no more than four and never more than, say, eight [^1].

<script src="https://d3js.org/d3.v5.js"></script>

<br>
<div id="categorical_example"> </div>
<br>

<script src="/docs/assets/javascript/palettes_introduction/catExample.js">
</script>

[^1]: There is no agreement on the exact number of maximum categories one can easily distinguish by using color. A recent article, however, suggested that we use two different areas of the brain to count up to four and to count more objects, so without other experimental evidences I would suggest four as possible maximum value.


## Quantitative color maps

The simplest possible way to construct a quantitative color map is to keep the hue and the chroma as constant, and to let the value vary within the largest possible range.
Many color scales have been built in this way, but this is not the only possible way to proceed.
A very popular and valid alternative is to vary hue too.
This has the main advantage that one may span a larger region of the color space, and this would increase the smaller possible perceived variation
of your attribute.

<br>
<div id="quantitative_example"> </div>
<br>

<script src="/docs/assets/javascript/palettes_introduction/quantExample.js">
</script>

## Diverging color maps
Building a diverging color map will now be very easy, as
it simply requires to attach two color maps.
Those color maps should:

- Have the same chroma
- Start from a common point
- Move in different hue directions
- Span the same distance in the luminance direction


<br>
<div id="diverging_example"> </div>
<br>

<script src="/docs/assets/javascript/palettes_introduction/divExample.js">
</script>

## Cyclic color maps

Sometimes you may also want to encode the fact that your attribute is cyclic, as an example when you are
dealing with months, seasons or other periodic quantities.
In this case you can simply put a difference between the minimum and maximum hue equal to 360.

In case you are dealing with categorical attributes, keep in mind than you should
increase by one the number of categories and drop the last one in order to 
have distinct colors for distinct categories.

<br>
<div id="cyclic_example"> </div>
<br>

<script src="/docs/assets/javascript/palettes_introduction/cclExample.js">
</script>

## Conclusions

Choosing the correct color map may be a complicated issue.
We have seen which are the main choices one must address in this task,
together with some recipes to proceed.
Sometimes the easiest choice is to simply go for a pre-build color map,
as most dataviz libraries comes with their palettes, and they are
usually done very well.
This is the case of [matplotlib](https://matplotlib.org/stable/users/explain/colors/colormaps.html) for Python or [ggplot2](http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/) for R, as well as
for other tools like [Tableau](https://help.tableau.com/current/pro/desktop/it-it/formatting_create_custom_colors.htm) and of course [d3js](https://observablehq.com/@d3/color-schemes) (which has been used almost everywhere in this blog, included this post).

Another great resource is to take a palette from [colorbrewer](https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3), which is a collection of hand made color maps.

Sometimes, however, this is not possible, since you
may have requirements which are not implemented into the most popular
palettes.

In any case, you should keep in mind the following truth about color:

<div class="emphbox">
<q>Above all, do not harm</q> - Edward Tufte
</div>


## Practice it 

Now that we highlighted some of the choices you should consider when building a
color map, let us use the above mentioned strategy to construct a color map.
This is a very simple app that allows you to implement the above methods
to construct a color map.
You can choose the chroma and select a segment in the Hue-Luminance plane.
One you are satisfied, you can generate your palette.
Notice that, by varying the parameters, you may get out from the displayable
region, so in order to avoid perceptual distortion the app will return black
for the points which are not displayable.


<br>

<div class='row' style="display:flex">
<div class='column' style="flex:50%;">

<div>

<br>
Categories: <input type='text' id='CategoricalNumClasses' />
<ul>
<li>
<input type='range' id='CategoricalChroma' min="0" max="120" onchange="updateChroma(this.value)" /> Chroma: <span id="chromaInput">45</span> 
</li>
<li> <input type='range' id='CategoricalminHue' min="-360" max="360" onchange="updateMinHue(this.value)" /> Hue 1: <span id="minHueInput">-70</span> 
</li>
<li><input type='range' id='CategoricalmaxHue' min="-360" max="360"  onchange="updateMaxHue(this.value)" /> Hue 2: <span id="maxHueInput">100</span>
</li>
<li><input type='range' id='CategoricalValueMin' min="0" max="100" onchange="updateMinValue(this.value)" /> Luminance 1: <span id="minValueInput" >15</span>
</li>
<li><input type='range' id='CategoricalValueMax' min="0" max="100" onchange="updateMaxValue(this.value)" /> Luminance 2: <span id="maxValueInput">95</span>
</li>
</ul>
</div>
<div>
<br>

 <button onclick="drawPalette()">Update palette</button> 
<button onclick='createFile()'>Download data</button>
</div>

<br>

 <div id="catPalette"></div>
 </div>
<div class='column' style="flex:50%;">
 <div id="catSurface"></div>
 </div>
 </div>

<br>
<br>
<div>
Here you may also import your palette to visualize it.
</div>
<div>
 <input type="text" id="inputPalette"/>
 <button onclick="drawExternalPalette()">Import color map</button> 

<br>
<br>

<br>
 <div id="externalPalette">
 </div>

<br>
<br>


<script src="/docs/assets/javascript/palettes_introduction/categorical_cmap.js">
</script>
