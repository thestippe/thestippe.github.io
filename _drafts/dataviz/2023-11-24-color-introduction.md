---
layout: post
title: "Introduction to color perception"
categories: /dataviz/
tags: /color-introduction/
image: "/docs/assets/images/color_perception/color_circle.png"
description: ""
---

Color is often used and misused in data visualization, as its stand out properties makes it 
a very attractive channel for any kind of visualization.
However, properly using color to encode any kind of attribute is far from being an easy
task.
Here we will start to analyze this problem by building a color space with the proper features.

## The RGB color space
<script src="https://d3js.org/d3.v7.js"></script>

As we have seen, we perceive color thanks to cones, which are divided
into three types. Each type is more sensitive to a specific wavelength,
corresponding to red, green and blue respectively.
This implies that, from a perceptual point of view,
color is a three dimensional space, as any color we see
can be defined by providing the amount of contribution of each of the three
channels.
These three channels are the same channels that are used by computers to specify
a colors: in the RGB space a color is defined by providing three numbers.
Since computers work in binary base rather than in base 10 base, the three numbers can span from $0$ to $255=2^8-1\,,$
and $0$ corresponds to no color in that channel, while $255$ means that 
that channel fully contributes to that color.
In the following you can visualize the mapping between the RGB input and the color output.


<br>
<br>
<div class='row' style="display:flex">
<div class='column' style="flex:20%;">
<div class="redContainer">
  <input type="range" min="0" max="255" value="0" class="slider" id="redRange">
  <p>R: <span id="redValue"></span></p>
</div>

<div class="greenContainer">
  <input type="range" min="0" max="255" value="0" class="slider" id="greenRange">
  <p>G: <span id="greenValue"></span></p>
</div>

<div class="blueContainer">
  <input type="range" min="0" max="255" value="0" class="slider" id="blueRange">
  <p>B: <span id="blueValue"></span></p>
</div>
</div>

<div class='column' style="flex:80%;">
<div class="svgContainer">
<svg height=120 width=100 id="rgbsvg">
<circle cx=50 cy=70 r=50 fill="rgb(0,0,0)" id="rgbCircle" stroke="black"/>
</svg>
</div>
</div>
</div>

<br>
<br>

## XYZ and Lab color spaces

Even if both the systems work by using red, green and blue as fundamental colors, the RGB color system does not
exactly correspond to how our eyes perceive the color, since the cones has a broad wavelength bandwidth.
In order to account for this, the International Commission on Illuminance (CIE, from the french acronym),
proposed in 1936 the XYZ color space, where the basis vectors X, Y and Z are given by a linear combination
of the R,G and B component.

While our eyes capture the R,G and B channels, this is not how the signal is elaborated by our brain.
According to the color opponency theory, defined in the literature as one of the most important models in color perception, we combine the perceived colors into three channels:

- Black-White
- Blue-Yellow
- Magenta-Green

This theory was born from the intuition that
we cannot perceive opponent colors simultaneously:
there is no reddish green neither bluish yellow.
For this reason, the CIE proposed the Lab color system, where the three coordinates are
the luminance $$L^*$$ (black-white channel) together with two chroma $$a^*$$, which goes spans from magenta to green,
and $$b^*$$, which runs in the blue-yellow direction.
The mapping between the XYZ color space to the Lab color space is non-linear in order for the Lab space
to be **perceptually uniform**, which means that a variation in this space
can be linearly mapped into a perceived color variation.
Another space with the same features is the Luv color space, but from the methodological point of view its 
construction is identical to the Lab color space.

## HCL color space

This color system is, however, nor very intuitive, as it is hard to specify a color by means of its $$a^*$$ and $$b^*$$
components.
This problem can, however, easily solved by switching to circular coordinates into the $$a-b$$ plane.
This color space, named LCh, has three coordinates:

- Hue (the angular coordinate)
- Chroma (the radial coordinate)
- Lightness

In the following we try and show the LCh color space.

<div class='row' style="display:flex">
<div class='column' style="flex:40%;">
<div id="colCircle">
</div>
</div>

<div class='column' style="flex:40%;">
<div class="luminanceContainer" style="margin-top:230px">
  <input type="range" min="0" max="100" value="50" class="slider" id="luminanceRange">
  <p>L: <span id="luminanceValue"></span></p>
</div>
</div>
</div>

<script src="/docs/assets/javascript/color_introduction/colorCircle.js">
</script>

Notice that the LCh color space can be both be constructed by starting from the Lab space and from the Luv space,
so a subscript in order to specify if the LCh color space has been build on the Lab color space or on the Luv color space.

<div class="emphbox">
The LCh color space is both perceptually uniform and easy to use.
</div>

There have been other attempts to map the RGB color space into more friendly color spaces, and the most
popular are HSL and HSV, but they are not perceptually uniform, so they are not well suited for data visualization.

## Conclusions

We saw how to switch from the RGB color space to
the HCL color space, going through the XYZ
and the Lab/Luv color spaces.
The HCL color space is perceptually
uniform, so variations into the color
space are linearly mapped to perceptual variations.
This feature makes this color space a fundamental
building block in data visualization, as it allows
us to use color to map quantitative attributes
in a non-deceptive way.
In a future post we will see how to use this
space to build color palettes.


<script src="/docs/assets/javascript/color_introduction/rgbshow.js">

</script>

