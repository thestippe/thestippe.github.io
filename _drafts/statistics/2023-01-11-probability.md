---
layout: post
title: "Probability foundations"
categories: /statistics/
tags: /probability
image: ""
description: "The axiomatic definition of probability"
---

In a [previous post](/statistics/statistics) we mentioned the axiomatic approach to probability.
Here we will discuss them and their most important consequences.

## The Kolmogorov axioms

In the context of probability theory, we consider an **experiment** any measurement
which can in principle be repeated an infinite amount of times.
The set of all the possible outcomes of the measure is called the **sample space** or **universe**,
and it is usually indicated with the letter $$\Omega$$.

If we want to describe the result of a coin toss, our sample space will be the set $$\{H, T\}\,,$$
while if we want to model the income of some population, the space will be the
set of the positive real numbers $$\mathbb{R}^+\,.$$

We then introduce the space $$F\,,$$ and the definition of this space is rather involved:
if we are dealing with a finite space, it is simply the collection of all the possible subsets of $$\Omega\,.$$
If it is not, however, things get more involved, and one must define it as a [**$$\sigma$$-algebra**](https://en.wikipedia.org/wiki/%CE%A3-algebra)
over $$\Omega$$.

So $$F$$ will be $$\{\emptyset, H, T,\{H,T\}\}$$ in the first case, while in the latter it will be the collection of all the finite segments
over the positive real axis.

We finally introduce the **probability function** $$P$$ which is a function $$P:F \rightarrow [0, 1]$$ such that:

- $P(A) \geq 0 \forall E \in F$ the probability must be non-negative
- $P(\Omega) = 1$
- $P\left(\bigcup_i E_i\right) = \sum_i P\left(E_i\right) \forall \{E_i\}  : E_i \cap E_j = \emptyset \forall i \neq j $ (the total probability of disjoint events is the sum of their probabilities)

## Some immediate consequences

Since, given any set $E$, we can decompose $$\Omega = E \cup \bar{E}$$ with $$E\cap\bar{E}=\emptyset$$
we immediately get $$P\left(\bar{E}\right)=1-P(E)$$ and,
obviously, $$P(\emptyset)=0$$.

Let us take $A\subseteq B\,,$ so $B = A \cup (B/A)$ and $A \cap (B/A) = \emptyset\,.$
We then have immediately $$P(B) = P(A) + P\left(B/A\right) \geq P(A)$$
so if $A \subseteq B \Rightarrow P(A) \leq P(B)\,.$

If we now take two arbitrary sets $A$ and $B\,,$ we can decompose
$$ A \cup B = A \cup (B/A) = A \cup (B/(A\cap B))$$
so $$P(A\cup B) = P(A) + P(B/(A\cap B))\,.$$
On the other hand we have that $$B=(B/(A\cap B)) \cup (A\cap B)$$
so $$P(B) = P(B/(A\cap B)) + P(A\cap B)$$ or equivalently $$P(B/(A\cap B)) = P(B)-P(A\cap B)\,.$$
By combining the two results we have

$$P(A\cup B) = P(A) + P(B) - P(A\cap B) $$

## Conditional probability

The conditional probability of $A$ given $B$ is defined as $$P(A | B) = \frac{P(A \cap B)}{P(B)}$$
Notice that, if both $P(A)>0$ and $P(B)>0\,,$ we can write

$$P(A \cap B) = P(B) P(A | B) = P(A) P(B | A)$$

which can be rewritten as

$$ P(A | B) = \frac{P(B|A) P(A)}{P(B)} $$

and this is the well known [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem).


## Conclusions

We formulated the Kolmogorov axioms, we showed some fundamental result of probability theory,
we defined the conditional probability and derived the Bayes' theorem.
In the future we will show how this theorem is used in Bayesian statistics to obtain informations about the parameters by using the data.


## Suggested readings

- <cite> <a href="https://books.google.it/books/about/The_Theory_of_Probability.html?id=g4sZAQAAIAAJ&redir_esc=y">Gnedenko (1978). The theory of probability.</a> </cite>
