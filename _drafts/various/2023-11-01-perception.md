---
layout: post
title: "Perception"
categories: course/various/
tags: /perception/
image: "/docs/assets/images/perception/eye.jpg"
description: "How do we see what surrounds us"
---

Before digging into visualizations, we must understand how do we perceive
images.
Typically the human eye can detect light in the range of wavelengths going
from 380 nanometers (violet) to 700 nanometers (red).
Our eye is rather complex, but to our aim we can only consider few components:
- The **cornea** which acts as a convergent lens and focuses the light coming to our eye.
- The **iris** which regulates the amount of light entering into our eye.
- The **crystalline lens** changes the focal length of the eye, allowing us to focus on different objects.
- The **retina** which contains the light receptors of our eye.
- The **optic nerve** which transmits the information from the retina to the brain.

Into the retina we have two kind of receptors:
- The **rod cells** which are very sensitive in low light conditions. They are roughly 90 millions and are especially used into the peripheral vision and night vision. They are especially concentrated at the outer edge of our retina
- The **cone cells** are responsible of the color vision, and are roughly 6 millions. Human eye has three kind of cone cells, and each type is more sensitive into a specific wavelengths range, corresponding approximately to **red**, **blue** and **green** wavelengths and named long, medium and short wavelengths cones.

The red cones are approximately ten times the green or blue ones, this is why we are better in discriminating the red tones than the blue or green ones.

![Cone absorbance](/docs/assets/images/perception/Cone-absorbance-en.svg) The typical spectrum of our light receptors, from [Wikipedia](https://en.wikipedia.org/wiki/Rod_cell)

The cone density is much higher in a small region located oppositely to the iris,
namely the **fovea**. Approximately half the nerve fibers in the optic nerve
carry information from the fovea, while the remaining half carry information
from the rest of the retina.

This should suggest you that the fovea is the region with the highest resolution
of the retina, and since it is very small our eye can only clearly see within
a very small region.
In fact, our high-resolution region is limited to less than 2 degrees.

On the other hand, on average, we have the feeling that we can clearly see most
of what surrounds us. This is because our eye makes small movements (less than 20 degrees)
named [**saccadic movements**](https://en.wikipedia.org/wiki/Saccade) with an average time between two movements of 225 ms.
Our brain then elaborates the images and reconstructs a map by using many movements,
giving us the feeling of a higher resolution.

This has a very important impact on data visualization: 

<div class="emphbox">
Attention plays  a major role into what we perceive.
</div>

## Preattentive features

For this reason the scientific community spent a lot of energy in trying and determine
what drives our attention.
According to Colin Ware's textbooks 
**Information Visualization: Perception for Design** the list of features which
drives our attention, named **preattentive features**, can be divided into
four kind of features: **form, color, motion** and **spatial positioning**.

**Form:**
- Line orientation
- Line length
- Line width
- Line collinearity
- Size
- Curvature
- Spatial grouping
- Blur
- Added marks
- Numerosity

**Color:**
- Color hue
- Color intensity

**Motion:**
- Flicker
- Direction of motion

**Spatial positioning:**
- 2D position
- Stereoscopic depth
- Convex/concave shape from shading

<div class="emphbox">
Whenever you want to encode a relevant information in your visualization,
you should use one of the above features.
</div>

Not all of them take the same amount of time to be processed.
As an example, it takes a very short time to recognize a red circle
between many blue circles.
If you don't trust me you can try and click on the red circle in the figure below.
Each time you will hit it a new figure will appear, and you will get visualize
the distribution of the time needed to hit it.

<!-- Load d3.js -->
<script src="https://d3js.org/d3.v5.js"></script>

<div id="preattentive_color"> </div>


<!-- Create a div where the graph will take place -->

You can now try and perform the same task, but this time you will be required
and hit the blue square.

<div id="preattentive_size"> </div>
<!--
 Create a div where the graph will take place -->


Properly using the preattentive features,
unfortunately, is far from being easy, especially when
you want to combine more than one feature.
Some feature combinations are still preattentive, while other are
no more preattentive, and at the best of my knowledge there are no
exhaustive studies where these combinations are analyzed.


## Bonus: the results of your test

### Color difference

<div id="preattentive_color_list"> </div>

### Size difference

<div id="preattentive_size_list"> </div>

<script src="/docs//assets/javascript/perception/preattentive_color.js"> </script>

<script src="/docs//assets/javascript/perception/preattentive_size.js"> </script>

