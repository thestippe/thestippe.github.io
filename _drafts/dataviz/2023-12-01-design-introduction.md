---
layout: post
title: "Design tricks"
categories: /dataviz/
tags: /design_introduction/
image: "/docs/assets/images/design.jpg"
description: "Few ideas from design"
---

Some day I was watching a video by Tamara Munzner,
where she recommended to get some basic knowledge about design.
Before dealing with data visualization I though that design
was a useless subject, but I decided to follow her suggestion.
I first looked for some book, and then I decided to attend a design course.
By doing so, I found many very useful suggestions, and I realized
that design is not about making your visualization beautiful,
but it's about making it clear.
Here I will share some of the information I found during these courses.

## Grid

Align, align everything! The more things are aligned, the easier will be
for your audience to find the informations.
Generally left alignment is better, but there are situations where
you may prefer to use other alignments.
A classical example is given by tables with one key and one value:
in this case you may want to right-align your left column
and left-align your right column.
By doing so, it will be easier for your reader to match each key with its own value.

| key | value |
|---:|:----|
| apples | 14 |
| oranges | 8 |
| bananas | 21 |

In the above table I used another trick to make the table easier to read, I alternated the background color of the rows.

A common error is to center everything, this may strongly affect the readability of your visualization.

## White space

Don't feel obliged to fill every squared centimeter of your visualization: leaving some white space may enhance its readability.
In other world, use the space you have at your disposal.
Moreover, white space in visual design has the same effect as pausing has when talking:

<br>
<br>
<br>
<br>
it adds emphasis!

## Consistency

Strive for consistency, make things look similar, and if you don't want to, make them look **very** different, and this is of course 
applies to any context.
Consistency will help your reader to understand you. If you once use an acronym, then you should always use it.

When you build a dashboard, make sure that your interacting items are build consistently.

Do not implement a light blue button and a dark blue one: you should either give them the same color if they have a similar function,
or you should give them a very different color, like blue and red.

Moreover, you should be consistent with the used conventions. Red in finance means loss, and if you're preparing a visualization
for a finance startup, then you should be consistent with this convention and reserve the red color for loss.

## Typography

Typography is by itself the topic of many books, blogs and other website, so I won't even try and be exhaustive here.
Just keep in mind that the choice of your font is very important for many factors:

- Font can affect the readability of your visualization
- A bad font choice can convey the wrong message
- You should always be consistent with the generally accepted typographic conventions

I will probably discuss typography more in detail in a future post, but for now keep in mind that this is a very important subject, unless your
visualization has no text (and in this case it would be hard to understand it).

- DON'T USE TOO MUCH UPPERCASE
- When you use commas , dots or other punctuation , space goes after , not before !
- When you quote, someone, like ''There's no place like home.'', you should use "double quotation marks"
- ***Too much emphasis is no emphasis.*** Avoid too much bold or italic, and avoid using them simultaneously
- em dash ---like this--- can be used instead of parenthesis, en dash can be used for dates, like 2023--12. Hyphen are used to join words or parts of words, like check-in
- If you write lists:
   - Either capitalize the first letter and use a dot.
   - or don't capitalize the first letter and use semicolon;
   - Or don't use punctuation
   - Don't capitalize with semicolon;

When you start a new paragraph, either enlarge space between rows, or add a space at the beginning of the new line.

 &nbsp; Don't do both.

## Color

We already discussed color from the perceptual point of view, but color is very important from the design perspective too.
Colors have cultural meaning, and being consistent with them is very important.
If you are developing a visualization about smog, you should keep in mind the fact that green can be often interpreted as natural or healthy,
so using green to depict a highly polluted area may not be a good idea.

You should also look for color combinations that look appealing, as they will let your reader look at your visualization with pleasure.

## Hierarchy

Keep in mind the hierarchy of your visualization, emphasize what is relevant and put less important objects in the background.
Find the focal point of your composition and try and determine the visual query that the reader will follow. Is this the one you look for?

