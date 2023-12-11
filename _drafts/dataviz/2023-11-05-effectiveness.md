---
layout: post
title: "Channel effectiveness"
categories: /dataviz/
tags: /effectiveness/
image: "/docs/assets/images/effectiveness/effectiveness.png"
description: "Quantifying the goodness of a channel to show an information"
---

<!-- Load d3.js -->
<script src="https://d3js.org/d3.v5.js"></script>

When we have been talking about [marks and channels](/marks-channels)
we mentioned the concept of effectiveness, by saying it
reflects how easily we translate the visual information encoded into the
corresponding channel.

Effectiveness is not a fundamental quantity, but it rather summarizes a
number of possible features:
- Accuracy
- Discriminability
- Salience
- Separability
- Grouping


In this post we will discuss the meaning of these terms, and we will 
see how to determine which channel is the most effective for your task.

## Accuracy
Accuracy quantifies how good is a channel in conveying the value of an attribute.
Not all channels are equally accurate, as we perceive different
channels in different ways.
Research showed that, on average, there is a power law relationship between the 
change in the stimulus and the perceived change, and this law goes under the name
of **Stevens' power law**

$$ \psi_k(I) \propto I^{a_k} $$

where $a_k$ is the exponent associated with the stimulus of type $k$,
$I$ is the intensity of the stimulus and $\psi$ represents the perceived
value.
Here we show the relation for some of the main channels we will use.

<div id="stevens"> </div>
<script src="/docs/assets/javascript/effectiveness/stevens.js"> </script>

As we see, the only quantity which we perceive linearly is the length,
while we are on average worst at estimating any other quantity.

As an example, try and estimate the length ratio between the two
lines and the area ratio between the two circles:

<br>
<br>

<svg height="150" width="600">
  <line x1="0" y1="40" x2="150" y2="40" style="stroke:crimson;stroke-width:10" />
  <line x1="0" y1="60" x2="450" y2="60" style="stroke:steelblue;stroke-width:10" />
</svg> 

<svg height="150" width="500">
  <circle cx="150" cy="50" r="28.87"  fill="crimson" />
  <circle cx="250" cy="50" r="50"  fill="steelblue" />
</svg> 

Both the ratios are equal to 3. Was it hard to do that? How accurate
have you been?

## Discriminability

Discriminability quantifies how many different values can we encode into
a certain channel by letting them being perceived differently.
Of course, this only becomes a problem as you approach the discriminability
limit of the channel.

Here we show 30 different tones of
red. Can we distinguish all of them? Honestly I think it's quite a hard task.

<div id="discriminability"> </div>
<script src="/docs/assets/javascript/effectiveness/discriminability.js"> </script>

## Salience

Salience tells us how easy it is for us to find differences among objects
by using a certain channel.
As we have previously seen, it's very easy to spot a red circle between
blue circles, so color hue has good salience capabilities.
Color luminance is much worst in this task, as it is very hard to spot
objects with different color luminance, so color luminance has 
worst salience (or worst popout properties) than color hue.

## Separability

Channels cannot be treated independently one on the others,
but the properties of one channels depend on the other channels
used in the visualization.
There are channels among this interaction is stronger, and those
channels are called **integral**, as well as channels where the interaction
is almost negligible, and they are called **separable**.


<svg height="150" width="500">
  <circle cy="50" cx="50" r="1.5"  fill="crimson" />
  <circle cy="50" cx="100" r="3"  fill="crimson" />
  <circle cy="50" cx="150" r="6"  fill="crimson" />
  <circle cy="50" cx="200" r="12"  fill="crimson" />
  <circle cy="50" cx="250" r="24"  fill="crimson" />
</svg> 


In the above figure, do you always perceive the same color? Or do
you rather think that the color of the ball changes with the circle?
Most of the people would say that the color changes among the circles,
and they would be wrong.
This is because color interacts with size, especially for small objects.

The interaction also goes the other way round:

<br>
<div style="background-color:black;">
<p style="color:red;font-size:60px;">
Most people perceive this as bigger</p>
<p style="color:blue;font-size:60px;">But the two
lines have the same size</p>
</div>

<br>
In the first case the color was affected by the size,
in the second case the other way round happened.

## Grouping

Grouping tells us how easy it is for us to spot patterns in the data.
In psychology it has been extensively studied what we perceive as grouped,
and these results are collected into the **Gestalt principles**. 

Gestalt principles are well known to whoever studied design, and we will
discuss them into a [separate post](/gestalt).

## Our perception depends on the context
What we perceive strongly depends
on the context.
As an example, the color perception of an object depends on the color of the
surrounding objects.

<br>
<br>

<svg height=400 width=1100>
<rect x=0 y=0 height=400 width=550 fill="#e7d645"/>
<rect x=550 y=0 height=400 width=550 fill="#8d9488"/>
<rect x=50 y=195 height=10 width=1000 fill="#b6af59"/>
</svg>

<br>
<br>

Would you always name the color of the above stripe in the same way?



## Conclusions
We have seen different criteria to assess
the effectiveness of a channel.
Depending on your task, you should find
the most appropriate way to assess the effectiveness
of a visualization.
If you want to precisely compare values, you
should probably favour more accurate channels,
while if you want to check if your
clustering algorithm is doing its job, then you should consider using channels where grouping is easier.
