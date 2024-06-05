---
layout: post
title: "Kolmogorov axioms"
categories: /statistics/
subcategory: Introduction
section: 1
tags: /kolmogorov/
date: "2024-01-06"
# image: "/docs/assets/images/perception/eye.jpg"
description: "The mathematical foundations of statistics"
published: false
---

In a previous post we mentioned the Kolmogorov axioms.
In this post we will discuss them and look at some immediate consequence
of these assumptions.
We will also discuss the definition of random variable,
and provide other important definition related to this concept.
We will assume that the reader already had a course about Lebesgue integrals
and multidimensional analysis, and we will only recall the basic definitions.


## The axioms of probability

$$\mathcal{F}$$ is a **$$\sigma$$-algebra** over $$\Omega$$ if it is a nonempty collection
of subsets of $$\Omega$$ such that
- $A \in \mathcal{F} \Rightarrow A^c \in \mathcal{F}$
- If $A_j$ is a succession in $\mathcal{F}$ then $\bigcup_j A_j \in \mathcal{F}$
- If $A_j$ is a succession in $\mathcal{F}$ then $\bigcap_j A_j \in \mathcal{F}$

$$P$$ is a **measure** over $$(\Omega, \mathcal{F})$$ if
- $P(A) \geq 0\, \forall A \in \mathcal{F}$
- $P(\emptyset) = 0 $
- If $A_j$ is a countable collection of disjoint elements of $\mathcal{F}\,,$ then $P\left(\bigcup_j A_j \right) = \sum_j P(A_j)$

$$(\Omega, \mathcal{F}, P)$$ is a **measure space** if
- $$\Omega$$ is a set
- $$\mathcal{F}$$ is a $$\sigma$$-algebra over $$\Omega$$
- $$P$$ is a measure over $$(\Omega, \mathcal{F})$$

We define a **probability space** is any measure space
$$(\Omega, \mathcal{F}, P)$$ such that $$P(\Omega) = 1\,.$$

## Consequences

$$1 = P(\Omega) = P(A \cup A^c) = P(A) + P(A^c)$$
so $P(A^c) = 1-P(A)\,.$
Since both the elements must be non-negative we have that $0 \leq P(A) \leq 1\,.$

If $A \subseteq B$ we can write $$B = A \cup (B \setminus A)$$ and $$A \cap (B \setminus A) = \emptyset\,,$$ so $$P(B) = P(A) + P(B \setminus A)\,.$$ Since both the elements of the RHS of the equation
must be non-negative we have that $$P(B) \geq P(A)\,.$$

\mathcal{F}or any elements $A, B \in \mathcal{F}$ we can write
$$A \cup B = A \cup (B \setminus A)$$ with $A \cap (B \setminus A) = \emptyset\,,$
so $$P(A\cup B) = P(A) + P(B\setminus A)\,.$$
We also have that $$B = (B \cap A) \cup (B \setminus A)$$
with $$(B \cap A) \cup (B \setminus A) = \emptyset$$ so
$$P(B) = P(B\cap A) + P(B\setminus A)$$ or $$P(B \setminus A) = P(B) - P(B\cap A)$$
and this implies

$$P(A \cup B) = P(A) + P(B) - P(A \cap B) $$

## Random variables

Working in $\Omega$ can be cumbersome, as we didn't required it to be equipped
with any operation, while we would like to be able to perform some computation.
For this reason it is useful to define a **real random variable**,
and this is defined as a measurable map
$$X : \Omega \rightarrow E \subset \mathbb{R}^n\,.$$
If $n=1$ we say $X$ is a scalar or univariate random variable,
if $n > 1$ we say it's a vector or multivariate
random variable, and $n$ is the dimension of the variable.
We define the **realization** of $X$ the value
taken by $X$ at $\omega \in \Omega\,,$
namely the value $x=X(\omega)\,.$

If $X$ is a one-dimensional random variable, we can define
$$F(x) = P(X \leq x)\,,$$ and $F$ is defined as the **cumulative distribution
function** (cdf for short) of $X\,.$
$F$ has the properties that
- $0 \leq F(x) \leq 1 \forall x \in E$
- $F(-\infty) = 0$
- $F(+\infty) = 1$

Moreover $P(x_1  < X \leq x_2) = F(x_2) - F(x_1)$

If $X$ is discrete, we can decompose
$$ F(x) = \sum_{x_j \leq x} p_X(x_j)$$ and $p_X$ is defined as the
**probability mass function** (pmf for short) of $X\,.$

If $X$ is continuous, we can decompose
$$ F(x) = \int_{y\leq x} dy p_X(y)$$ and $p_X$ is defined as the
**probability distribution function** (pdf for short) of $X\,.$

In the following we will often omit the subscript $X$ and we will indicate
the pdf/pmf as $p\,.$

We can immediately generalize to the $n$ dimensional case,
where $X = (X_1,\dots,X_n)$, and define
$F(x_1,\dots,x_n) = P(X_1 \leq x_1,\dots X_n \leq x_n)\,.$

In the discrete case the pmf generalizes to
$$F(x_1,\dots,x_n) = \sum_{y_1 \leq x_1,\dots,y_n \leq x_n} p(y_1,\dots,y_n)\,,$$
while in the continuous one we have
$$F(x_1,\dots,x_n) = \int_{y_j \leq x_j} \prod_j dy_j p(y_1,\dots,y_n)\,.$$


If $X$ is a continuous random variable, we define the **expected value** of any measurable function $f$
as

$$\mathbb{E}[f(X)] = \int_E p(x) f(x)$$

In the discrete case we simply replace the integral with the sum.

We define the **mean** as
$$\mathbb{E}[X] = \int dx x p(x)$$

If $X$ is a one dimensional random variable, we define the **variance** as
$$ \mathbb{E}[(X-\mathbb{E}[X])^2]\,, $$
while the $k$-th raw moment is defined as
$$ \mathbb{E}[X^k]\,. $$
Notice that, in general, there is no warranty that moments of order
$k>0$ exists.

On the other hand, if $X$ is multidimensional, we define the component $(i, j)$ of the  **covariance matrix**
as
$$\mathbb{E}[(X_i - \mathbb{E}[X_i])(X_j - \mathbb{E}[X_j])]\,.$$

The **correlation** between $X_i $ and $X_j$ is defined as

$$
\frac{
\mathbb{E}[(X_i -\mathbb{E}[X_i])(X_j - \mathbb{E}(X_j))]
}{\sqrt{ \mathbb{E}(X_i - \mathbb{E}[X_i])^2 \mathbb{E}(X_j - \mathbb{E}[X_j])^2  }}
$$

Notice that we can write
$$
F(x_1,\dots, x_n) = \mathbb{E}[\theta_{x_1}(X_1)\dots \theta_{x_n}(X_n)]
$$
where $\theta_x$ is the Heaviside step function

$$
\theta_x(y) =
\begin{cases}
& 1 & y \leq x\\
& 0 & y > x \\
\end{cases}
\,.
$$ 

We define the **characteristic function**

$$
\varphi(t) = \mathbb{E}[e^{i t \cdot X}]
$$

If $X$ is a one dimensional variable, assuming that all the moments are finite, we have that
$$
\varphi(t) = \sum_{j=0}^\infty i^j \frac{\mathbb{E}[X^j] t^j}{j!}\,.
$$
so we can recover any moment as a derivative of the characteristic function.

Another very important expected value is the **entropy**

$$\mathbb{E}[-\log(p(X))]$$

If $X$ is an $n$ dimensional continuous random variable and $Y = g(X) = (g_1(X),\dots, g_n(X))$
is an invertible transformation with inverse
$h = (h_1,\dots, h_n)$, then

$$
p_Y(y) = p_X(h(y)) |J|
$$

where 

$$ J = \det \left(\frac{\partial h_i(y)}{\partial y_j} \right)
$$


## Joint probabilities

Given a multivariate distribution $p(x_1,\dots x_j,\dots,x_n)$
we define the **marginal** pdf of $x_j$ as

$$p(x_j) = \int \prod_{i \neq j} dx_i p(x_1,\dots x_j, \dots,x_n)\,.$$
Notice that $p(x_j)$ defines a probability density function,
as it is non-negative and in integrates to 1.

We can also write $p(x_j)$ as
$$ p(x_j) = \mathbb{E}[\delta(X_j-x_j)] $$
where $\delta$ is the Dirac delta distribution.

Of course, the extension of the marginal pdf to more than one dimension
is straightforward.

Given two random variables $X$ and $Y\,,$ we stay that they are
**independent** if $$p_{X, Y}(x, y) = p_X(x) p_Y(y)\,.$$
Notice that, if two variables are independent, we can factor any expected
value.

If $Z = (X, Y)\,,$ we define $p(z) = p(x, y)$ as the **joint** probability
distribution of $X$ and $Y\,.$
Given a multivariate random variable $Z=(X, Y)\,,$ with joint pdf $p(z) = p(x, y)$ we define the **conditional** pdf of $X$ with respect to $Y$ as

$$p(x | y) = \frac{p(x, y)}{p(y)}$$

where $p(y) = \int dx p(x, y)$ is the marginal pdf of $y\,.$
We stress that $$p(x | y)$$ defines a pdf for $x\,,$ since
$$ \int dx p(x | y) = \int dx \frac{p(x, y)}{p(y)} = \frac{p(y)}{p(y)} = 1\,, $$
but it doesn't define
a probability for $y\,,$ as $\int dy p(x | y) \neq 1\,.$

Notice that, since
$$ p(x | y) = \frac{p(x, y)}{p(y)}$$ and $$p(y | x) = \frac{p(x, y)}{p(x)}\,, $$
we can write

$$
p(x | y) = \frac{p(x | y)}{p(y)} p(x) 
$$

and this is the **Bayes' theorem** for the pdf.

Given a set of $n$ independent random variables with pdf $p_1,\dots,p_n\,,$
we say that they are **identically distributed** if

$$p_i(x)  = p_j(x) \forall i, j$$

and in this case we can write
$$p(x_1,\dots,x_n) = \prod_{i=1}^n p_1(x_i)\,. $$

## Conclusions

We defined a probability space and we showed some basic results
of the definition of probability.
We also gave the definition of random variable, and discussed some
basic concept in probability theory.
