---
layout: post
title: "Data classification"
categories: /dataviz/
tags: /data-types/
description: "An introduction to data abstraction"
---

Data can come into many different flavors, and in this post we will take a look at how
we can classify them depending on their qualities and on our previous knowledge.

In the simplest case our dataset will consist into a collection of
**items**, and each item will have a set of **attribute**.

In this case we are dealing with a **tabular** dataset.
A more involved case is the one we also have **relationships** between the
items, and in this case we are working with a **network**.

For the moment we won't talk about networks, and we will focus on tabular datasets.
A tabular dataset can be represented, as the name suggests, with a table:

| item | attribute 1 | attribute 2 | ... | attribute N |
|------|-------------|-------------|-----|-------------|
| 1 | $a_1$ | $b_1$ | ... | $z_1$ |
| 2 | $a_2$ | $b_2$ | ... | $z_2$ |
| 3 | $a_3$ | $b_3$ | ... | $z_3$ |
| ... | ... | ... | ... | ... |
| M | $a_M$ | $b_M$ | ... | $z_M$ |

Each attribute can be classified in a large number of ways, and here we will only
discuss the main ones.

## Attribute types

Attribute types define the mathematical operations that makes sense to do
on an attribute.

If an attribute doesn't have a natural order we call it **categorical**
or **nominal**.

Examples of nominal attributes are
- City names
- Gender (assuming it's a discrete quantity).
- Eye color

If an attribute has a natural order but we can't define
a distance between their values we say it's **ordinal**.

Ordinal attributes are
- Grade
- Education level
- Sport ranking

In this case, it doesn't makes sense to sum, subtract or do any other
arithmetic operations over the attribute.

If our attribute comes with a concept of distance we have
**quantitative** attributes.

Quantitative attributes can be further sub-classified depending if they
have a natural zero or not.
Attributes which don't have a natural zero are called **interval** attributes,
while those which have are named **ratio** attributes.

Example of interval attributes are

- Temperatures expressed in Fahrenheit or Celsius degrees
- pH

While ratio attributes are:

- Temperatures expressed in Kelvin degrees
- Length
- Mass
- any percentage
- Earnings (where negative earning means loss)

## Attribute semantics

The mathematical operations we can perform with an attribute
isn't of course the only meaningful way we can classify the data.
Another very important aspect of an attribute that we should
consider when choosing how to visualize an attribute is its meaning.

| Semantic | Meaning | Example |
|----------|---------|---------|
| Spatial | Our attribute has a geographical connotation | zip code, latitude, city name |
| Temporal | Our attribute is associated with time | seconds, day of the week |
| Hierarchical | Two or more attributes have a natural hierarchy | year, month, day, hour but also continent, country, city |

Moreover we can also classify our attribute depending if it is periodic or not,
if it is we say it is **seasonal**, **periodic** or **cyclic**.
Example of cyclic attributes are seasons or months.

## Conclusions

Before deciding how to visualize your dataset you must first of all consider
what are the types of the attribute you want to visualize and
whether they have some special connotation.
Here we have seen how to determine the attribute type and the most
common attribute semantic.
In the next post we will discuss some of the most common visualizations
associated with each data type.
