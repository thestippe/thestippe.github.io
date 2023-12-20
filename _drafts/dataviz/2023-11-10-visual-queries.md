---
layout: post
title: "Visual queries"
categories: /dataviz/
tags: /gestalt/
image: ""
description: "How do we inspect an image"
---

As we have seen, our eyes can only perceive a very small area with a high resolution. 
The saccadic movements, however, let us inspect different portions
of space. Our brain then reconstructs the information, 
making us confident that we clearly see a larger amount of space.

The sequence of inspected areas is called visual queries, and in this post
we will have a deeper discussion on them.

## A simplified information flow model
As explained in Colin Ware's textbook, the neural activity involved in our sight process is very involved, but
for our purposes we can build a simplified model, which will help us in designing visualizations.
In this model we have two information flows, a bottom-up flow and a top-down one.
In the bottom-up flow the information provided by our eyes is collected, filtered and elaborated.
The elaboration goes through many steps. In the first step, the retinal signal is converted into features like color, size and shape.
At each subsequent step we construct pattern of increasing complexity.
Some of the most relevant patterns emerging in this phase will be discussed in the post about [Gestalt psychology](/gestalt/).
These patterns are then stored into our **visual working memory**.

The visual working memory is considered the main bottleneck of our attention, as it can only store few objects at each time
and only for a very small amount of time, typically few seconds or less.
For this reason, it is generally recommended not to use interaction whenever possible, as interacting takes time, and during
this time some important information may get lost, so we may miss some important pattern [^1].




In the top-down flow we direct our attention, so our eyes, depending on our available information and expectations, as well as depending on our task.
At the lower level, our attention makes us focus on the elementary signals we are looking for: if we are looking for our car, and our car is red,
our attention will let us focus on red objects.
I would like to stress you with the fact that signals considered as interesting are stronger have a very important consequence:

<div class='emphbox'>
Our perception is strongly biased by what we expect.
</div>

There is also another important aspect that enters into the game, which is our culture, and this fact is well known by user experience designers.
As an example, our visual queries are generally influenced by our writing system: in most western cultures we write from left to right
and from top to bottom, and this makes us start looking for new information at the top left of a web page, so UI designers usually put
there the most important informations.
On the other hand, in other Countries like the ones in the Middle East, the writing goes from right to left.
Due to this, companies like Google or Microsoft usually reverse the user interface on the left-right plane for languages like arabic, and they put
the most relevant information on the right.


## Conclusions

Since the exact structure of the query is subjective and depends on the task, there is no recipe to determine how one will perform
visual queries.
There are however few questions that you can ask yourself when making a visualization:

- What task should the visualization accomplish?
- Who will be the audience?
- What could be a reasonable way to perform this task from a cognitive point of view?
- Is there any visualization which could make this task easier, from a cognitive point of view?

[^1]: As a fun fact, this memory bottleneck is used in many magic tricks, as the one [here](https://www.crazycardtrick.com/).
