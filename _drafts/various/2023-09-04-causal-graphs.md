---
layout: post
title: "Introduction to causal graphs"
categories: course/various/
tags: /causal-graphs/
image: "/docs/assets/images/causal_graphs/covariates.png"
description: "Representing causality flows"
---

Causality is a relation between events, and one of the easiest to interpret
representations of items and relations is by using graphs.
In our discussion we will roughly follow (in a slightly less rigorous and precise way) chapter 3 of [Brady Neal's notes](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf).
Since our relation, causality, is directional (either $A$ causes $B$ or $B$ causes $A$)
we must use directed graph or digraphs.

Formally a digraph $\mathcal{G}$ is a pair $(V, E)$ where $V$ represents
the collection of the vertices 

$$V=\{x_1, x_2,...,x_N\}$$

The elements of $V$ are also called nodes or points.

$E$ is the collection of the edges

$$E \subseteq \{(x, y) \in V \times V | x \neq y\}$$

We must make a further requirement: we must forbid cases where
$A$ causes $B$, $B$ causes $C$ and $C$ causes $A$, as it doesn't make sense
to have a circular causality relation. We thus must represent the causality
relation as a **Directed Acyclic Graph** or DAG.

Of course, if we only have 2 elements, either there is a relation between them
or there isn't, so the only possible set of edges between $x$ and $y$
are 

$$\{\}$$

$$\{(x, y)\}$$

$$\{(y, x)\}$$

In our representation, we assume that **every parent is direct cause of its children**,
so in the first case we have that $x$ and $y$ are independent,
in the second one we have that $x$ causes $y$, while in the last one we have that
$y$ causes $x$.

In terms of probabilities, we have that the probability can be represented
 as $p(x)p(y)$, $p(x)p(y \vert x)$ or $p(y)p(x \vert y)$ respectively.

## Building blocks

Let us take a look at what we might have with three vertices and two arrows:

The first possible case is called the **chain**, such that $p(x, y, z)=p(x)p(y\vert x)p(z\vert y)$

{:refdef: style="text-align: center;"}
![The chain](/docs/assets/images/causal_graphs/test-1.svg){: width="450" }
{: refdef}

We then have the **fork**, where $p(x, y, z) = p(y) p(x \vert y) p(z \vert y)$

{:refdef: style="text-align: center;"}
![The chain](/docs/assets/images/causal_graphs/test-2.svg){: width="280" }
{: refdef}

And we finally have the **immorality** $p(x, y, z) = p(x)p(z)p(y\vert x, z)$

{:refdef: style="text-align: center;"}
![The chain](/docs/assets/images/causal_graphs/test-3.svg){: width="280" }
{: refdef}

## Association flow

Let us check the flow of association for the three graphs.
For the chain we have that, when $x$ changes we have a change in $y$, and a change
in $y$ causes a change in $z$, so we will see some association flow between $x$ and $z$.

For the fork, analogously, we have that a change in $y$ will cause both
a change in $x$ and in $z$, so they will change together and we will see some
flow of association.

Vice versa, in the case of immorality, $x$ and $z$ will vary in a totally independent
way, and we will generally not see an association between them.

Let us see this mathematically.

For the chain:

$$
\begin{align}
&
p(x, y, z) = p(x) p(y\vert x) p(z \vert y)
\\
&
p(x, z) = \int dy p(x, y, z) = p(x) \int dy p(y \vert x) p(z \vert y) = p(x) p(z \vert x) \neq p(x)p(z)
\end{align}
$$

So if we simply ignore the intermediate variable $y$, we see an association between $x$ and $z$.
Analogously, for the fork:

$$
\begin{align}
&
p(x, y, z) = p(y) p(x\vert y) p(z \vert y)
\\
&
p(x, z) = \int dy p(x, y, z) = \int dy p(y) p(x \vert y) p(z \vert y)  = \int dy p(x) p(y \vert x) p(z \vert y) = p(x) \int dy p(y \vert x) p(z \vert y) = p(x) p(z \vert x) \neq p(x) p(z)
\end{align}
$$

For the immorality, on the other hand:

$$
\begin{align}
&
p(x, y, z) = p(x) p(z) p(y \vert x, z)
\\
&
p(x, z) = \int dy  p(x) p(z) p(y \vert x, z) =  p(x) p(z) \int dy p(y \vert x, z) = p(x) p(z)
\end{align}
$$

## Blocking

Let us now see what happens when we block for (or control) $y$:

$$
\begin{align}
&
p(x, y, z) = p(x) p(y\vert x) p(z \vert y)
\\
&
p(x, z | y) = \frac{ p(x) p(y \vert x)  }{p(y)} p(z \vert y) = p(x \vert y) p(z \vert y)
\end{align}
$$

The last equality follows from Bayes theorem $p(y \vert x) p(x) = p(x \vert y) p(y)\,.$
We can now prove unconfoundedness:

$$
p(z \vert x, y) = \frac{p(x, z \vert y)}{p(x \vert y)} = p(z \vert y)
$$

Let us now check the same for the fork:


$$
\begin{align}
&
p(x, y, z) = p(y) p(x\vert y) p(z \vert y)
\\
&
p(x, z \vert y) = \frac{p(x, y, z)}{p(y)} = p(x \vert y) p(z \vert y)
\\
&
p(z \vert x, y) = \frac{p(x, z \vert y)}{p(x \vert y)} = p(z \vert y)
\end{align}
$$

This cannot be done for the immorality:

$$
\begin{align}
&
p(x, y, z) = p(x) p(z) p(y\vert x, z)
\\
&
p(x, z \vert y) = p(x) \frac{p(y \vert x, z)p(z)}{p(y)} = p(z) p(z\vert x, y)
\end{align}
$$

We can now compute 

$$
p(z \vert y) = \int dx p(x, z \vert y) = p(z) \int dx p(z \vert x, y) \neq p(z \vert x, y)
$$

This implies that controlling for an immorality introduces an association between
the variables.

## Types of paths and d-separation

We can now consider an arbitrary path $$\{x_1, x_2, ..., x_N\}$$,
where $x_2,...,x_{N-1}$ are the central vertices of either forks or chains or
inverted chains.
We say that the path is **d-separated** by a set of blocked nodes $C$
if
- the path contains a chain (or fork), and the middle vertex of the chain (fork) is in $C$
- or if the path contains an inverted chain, and nor the middle vertex of the inverted fork $Z$ neither any descendant of $Z$ is in $C$.

We say that two vertices $A$ and $B$ are **blocked** by a set of nodes $C$
if all the paths from $A$ to $B$ are d-separated by $C$.
