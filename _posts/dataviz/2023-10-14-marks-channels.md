---
layout: post
title: "Marks and channels"
categories: /dataviz/
tags: /marks-channels/
image: "/docs/assets/images/markers_channels/markers.png"
description: "The building blocks of data visualization"
---

When we build a data visualization we are building vocabulary
to translate our data into a message, and this vocabulary can be decomposed in 
a certain number of elements.
We always do that, even if you may not conscious about it.
Being aware of this helps us to think about how to better build this vocabulary
and, ultimately, how to make your message more effective.

Here we will discuss these components, together with the
fundamental design principles of data visualization, which are a set of 
guidelines to help us finding the most appropriate representation for our data.

## Marks and channels

Marks and channels are the first two fundamental building blocks of a
data visualization.

### Marks
Marks are the geometric elements that we use to identify the items
of our dataset [^1].
Since we are plotting on a two dimensional surface, we can use any geometric entity
with dimensionality less or equal to two to represent our items, so we can use:

<svg height="180" width="95%" style="margin:2%; font-weight: bold">
<text x="30" y="15"> POINTS </text>
  <circle cx="20" cy="40" r="10" fill="steelblue" />
  <circle cx="45" cy="45" r="10" fill="steelblue" />
  <circle cx="70" cy="55" r="10" fill="steelblue" />
  <rect x="100" y="25" width="20" height=20 fill="steelblue" />
  <rect x="100" y="50" width="20" height=20 fill="steelblue" />

<text x="315" y="15"> LINES </text>
<line x1="270" y1="40" x2="420" y2="40" style="color:black; width:20px; stroke:black; stroke-width:3px"/>
<line x1="270" y1="80" x2="420" y2="60" style="color:black; width:20px; stroke:black; stroke-width:3px"/>
<line x1="270" y1="100" x2="420" y2="120" style="color:black; width:20px; stroke:black; stroke-width:3px"/>


<text x="630" y="15"> AREAS </text>
<rect x="600" y="25" width="120" height=80 fill="steelblue" style="stroke:black; stroke-width:3px" />
<rect x="600" y="25" width="80" height=80 fill="steelblue" style="stroke:black; stroke-width:3px" />
<rect x="600" y="25" width="80" height=40 fill="steelblue" style="stroke:black; stroke-width:3px" />
</svg> 

<div class="emphbox">
Someone also uses three-dimensional objects as markers.
Please, don't! 
<br>
Their encoding implies that we must model the perspective,
and this causes a distortion into our perceived values, making them not suitable
for data visualization.
</div>

### Channels
We then have the channels, and they encode the values of the data associated to our items.

The most commonly channels used to encode quantitative information are

| Channel | Example |
|---------:|:---------|
| Position on aligned axis | <svg height="50" width="200"><line x1="10" y1="10" x2="200" y2="10" style="color:black; width:20px; stroke:black; stroke-width:3px"/> <circle cx="45" cy="10" r="10" fill="grey" /> <line x1="10" y1="30" x2="200" y2="30" style="color:black; width:20px; stroke:black; stroke-width:3px"/> <circle cx="125" cy="30" r="10" fill="grey" /></svg>  | 
| Position on unaligned axis | <svg height="50" width="200"><line x1="10" y1="10" x2="160" y2="10" style="color:black; width:20px; stroke:black; stroke-width:3px"/> <circle cx="45" cy="10" r="10" fill="grey" /> <line x1="50" y1="30" x2="200" y2="30" style="color:black; width:20px; stroke:black; stroke-width:3px"/> <circle cx="125" cy="30" r="10" fill="grey" /></svg>  |
| Length | <svg height="50" width="200"><line x1="10" y1="10" x2="200" y2="10" style="color:black; width:20px; stroke:black; stroke-width:3px"/>  <line x1="10" y1="30" x2="150" y2="30" style="color:black; width:20px; stroke:black; stroke-width:3px"/> </svg>  | 
| Width | <svg height="50" width="200"><line x1="10" y1="10" x2="200" y2="10" style="color:black; width:20px; stroke:black; stroke-width:3px"/>  <line x1="10" y1="30" x2="200" y2="30" style="color:black; width:30px; stroke:black; stroke-width:5px"/> </svg>  | 
| Angle/Slope | <svg height="50" width="200"><line x1="10" y1="10" x2="70" y2="10" style="color:black; width:20px; stroke:black; stroke-width:3px"/>  <line x1="10" y1="10" x2="70" y2="30" style="color:black; width:20px; stroke:black; stroke-width:3px"/> <line x1="150" y1="10" x2="200" y2="10" style="color:black; width:20px; stroke:black; stroke-width:3px"/>  <line x1="150" y1="10" x2="200" y2="40" style="color:black; width:20px; stroke:black; stroke-width:3px"/></svg>  | 
| Area | <svg height="50" width="200"> <circle cx="20" cy="20" r="10" fill="grey" /> <circle cx="100" cy="20" r="15" fill="grey" /> <circle cx="180" cy="20" r="20" fill="grey" /> </svg>  | 
| Color luminance | <svg height="50" width="200"> <rect x="5" y="5" height="40" width="40" fill="lightgray"/> <rect x="85" y="5" height="40" width="40" fill="gray"/><rect x="160" y="5" height="40" width="40" fill="black"/>  </svg>  | 
| Color saturation | <svg height="50" width="200"> <rect x="5" y="5" height="40" width="40" fill="#61679e"/> <rect x="85" y="5" height="40" width="40" fill="#3644c9"/><rect x="160" y="5" height="40" width="40" fill="#0019ff"/>  </svg>  | 


On the other hand, if we want to encode a categorical information, we have the following
channels:

| Channel | Example |
|---------:|:---------|
| Spatial region | <svg height="60" width="200"> <rect x="5" y="5" height="40" width="40" fill="gray"/> <rect x="85" y="15" height="40" width="40" fill="gray"/><rect x="160" y="5" height="30" width="30" fill="gray"/>  </svg>  | 
| Color hue | <svg height="50" width="200"> <rect x="5" y="5" height="40" width="40" fill="#a31919"/> <rect x="85" y="5" height="40" width="40" fill="#19a319"/><rect x="160" y="5" height="40" width="40" fill="#1919a3"/>  </svg>  | 
| Shape/Texture | <svg height="50" width="200"> <rect x="5" y="5" height="40" width="40" fill="grey"/> <circle cx="105" cy="25" r="20" fill="grey"/> <polygon points="180,45 160,5 200,5" fill="grey" /> </svg>  | 

The order of the items in the above tables is not random, but it reflects how easily we
translate the visual information either into a quantity or into different categories,
and this property is called **effectiveness**.

### Other components

Marks and channels are the components which encode information about our data, but
they are not the only components which constitute a data visualization.
Other components are the ones which allow us to interpret, compare and give context
to our data.

- axis
- grids
- annotations
- legends
- labels 
- ticks
- reference lines

<div class="emphbox"> Annotations may also be used to draw attention to patterns of interest.</div>

Use this components only if they really help your reader.
Remember that, most of the time, you want the reader to easily compare the values of your data, not to be able to assess the exact value of
your data.
You should keep your visualization as clean as possible, and in order to do so:

- Avoid useless boxes
- Don't use grids if they don't help understanding the data
- Avoid too many ticks [^2]

## Design principles

The above mentioned concept should be applied, as much as possible, according to the
principles of data visualization design.
These principles, that we will discuss in the rest of this post, will allow the reader
to easily understand and decode the visualization, reducing the risk of a misinterpretation and so making the communication between you and your reader clearer.

### Expressiveness principle
The expressiveness principle essentially states that we should make the message
as clear as possible, without neglecting information and, probably most important,
without adding, either implicitly or explicitly, information.

<div class=emphbox>
The visual representation should represent all and
only the relations that exist in the data.
</div>

Some examples of violation to this principle are:
- Line chart used to represent categorical data
- Different color intensities to encode different categories
- A diverging color map to represent a quantity that does not have an origin (a zero).
- Using a channel that don't encode any data.



### Effectiveness principle
The effectiveness principle is another principle that
helps us finding the most appropriate channel for each variable.

<div class=emphbox>
The relevance of the information should match the effectiveness of the channel.
</div>

The first obvious consequence of this principle is that the most important variable
should always be encoded by using a spatial dimension.
On the other hand, one should not rely on color to effectively communicate
relevant quantities.

## Conclusions

We have first seen what are the main components of any data visualization.
We have then mentioned two important recommendations on how to translate our dataset into
a graph.

You should always keep them in mind, but you should also balance them by taking
into account your audience and your message.

## Suggested readings

<cite> Munzner, T. (2015). Visualization Analysis and Design. CRC Press. ISBN: 9781498759717 </cite>


[^1]: If we are representing a network rather than a tabular dataset, they are used to represents the links too.
[^2]: The ticks are the graphical components that are used to mark the values of the axis.
