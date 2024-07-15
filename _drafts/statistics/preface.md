---
layout: post
title: "An overview to statistics"
categories: /statistics/
subcategory: Introduction
tags: /statistics_overview/
date: "2023-11-01"
# image: "/docs/assets/images/perception/eye.jpg"
description: "What well we talk about"
section: 0
---

Before we start with the technical staff, we should clarify what well we talk about.
Since most of this section will be devoted to statistics, it is better to clarify what do we mean with 
statistics.

Most of the definitions of statistics are very similar to the following one

<br>

> Statistics is the discipline that concerns the collection, organization,
> analysis, interpretation, and presentation of data.
> 
> Wikipedia

<br>

This is a good definition, but we may also define statistics as the profession performed
by statisticians. In this case the question is: **what do statisticians do?**

One of my favourite answers to this question has been given by the
Lawrence Livermore National Laboratory statistician Kristin Lennox in [this video](https://www.youtube.com/watch?v=eDMGDhyDxuY&t=826s),
in what she defines as the "Central Dogma of Statistics"

<br>

> Statisticians use probability to quantify uncertainty.
> 
> Kristin Lennox

<br>

There have been proposed many reasons to justify why probability is the most appropriate tool
to quantify uncertainties, but the most convincing one is that probability
is the best tool we have up to now to do so.
In this part of the blog, we will discuss exactly this topic, namely how to use **probability**
to handle **uncertainties**.

## Probability and uncertainty

In the previous section we introduced two concept: probability and uncertainty.
If you are not familiar with statistics, you may wonder what do they exactly are.

### Probability

Probability is a mathematical concept which has been formalized by the mathematician [A. N. Kolmogorov](
https://it.wikipedia.org/wiki/Andrej_Nikolaevi%C4%8D_Kolmogorov).
If we are working onto a discrete space $\Omega\,,$
we may define a probability on it by defining any function $p : \Omega \rightarrow [0, 1]$ such that

- $ p(\omega) \geq 0 \forall \omega \in \Omega $
- $ p(\Omega) = 1$
- If $A_j \subseteq \Omega\,, j=1,2,...$ and $A_i \cap A_j = \emptyset \forall i \neq j\,,$ then $p(\cup_i A_i) = \sum_i p(A_i)\,.$

For the continuous case, things are a little bit more involved, but what we are interested in is that
in order to define a probability on any space $\Omega$ we only need to define a **probability
density function** $p$ such that 
- $p(x) \geq 0 \forall x \in \Omega$
- $\int_\Omega dx p(x) = 1\,.$

In this case we can define a probability on any subset $A \subseteq \Omega$ as 

$$P(A) = \int_A dx p(x)\,.$$

In the discrete case, we define a **probability mass function** $p$ and replace the integral with the sum

$$ P(A) = \sum_{\omega_j \in A} p(\omega_j) \,.$$

Here and in the following I assume you are familiar with the concept of integral.

### Uncertainty

There is now another concept which we should clarify, and this is uncertainty.
As Aki Vehtari explains in [his course](https://www.youtube.com/watch?v=AcKRob0C8EY&list=PLBqnAso5Dy7O0IVoVn2b-WtetXQk5CDk6),
philosophers distinguish between two types of uncertainty:
- **Aleatoric** uncertainty, which is due to an intrinsic randomness of the process under study.
- **Epistemic** uncertainty, which is originated by our ignorance.

What we should try, is to properly quantify both types of uncertainties, and use empirical observations to reduce
the epistemic uncertainty as much as we can.
In other words, as explained by [prof. Tony O'Hagan](http://www.stat.columbia.edu/~gelman/stuff_for_blog/ohagan.pdf) 
on the [Significance journal](https://academic.oup.com/jrssig?login=false).

<br>

> The whole purpose of Statistics is to learn
> from data, so there is epistemic uncertainty
> in all statistical problems. The uncertainty
> in the data themselves is both aleatory, because they are subject to random sampling
> or observation errors, and epistemic, because there are always unknown parameters
> to learn about.
> 
> Tony O’Hagan, Significance

<br>

## Conclusions

We explained what is statistics, namely the science which quantifies uncertainties by using probability,
and we defined these concepts.
In the next post we will take a closer look to the different approaches one can use to
quantify uncertainty, and as we will see each approach will lead to a different statistics subfield.