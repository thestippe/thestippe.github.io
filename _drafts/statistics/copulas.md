---
layout: post
title: "The copula model"
categories: /statistics/
subcategory: "Other random thoughts"
tags: /mrp/
date: "2024-09-23"
section: 9
# image: "/docs/assets/images/perception/eye.jpg"
description: "Building more flexible multivariate distribution from simpler ones"
---

Copulas are a very popular family of distribution, especially in finance.
In the past years, I have been using them for modelling portfolio risk.
This topic has already been discussed by the PyMC team in
[this page](https://www.pymc.io/projects/examples/en/latest/howto/copula-estimation.html),
and I strongly recommend you to take a look at that page.

Their approach is great from the point of view of the didactics, but
it is not the approach I would have used in my own project.
I therefore decided to implement my own version of a copula distribution,
and this has been a great opportunity to understand how to write a custom
multivariate distribution from scratch in PyMC.

## Main ideas behind the copulas.

Consider an arbitrary multivariate distribution, which has joint
cumulative distribution function

$$
F(x) = F(x_1,...,x_n)
$$

Let us now define the $i$-th cumulative marginal distribution as $F_i(x_i)\,.$
We already know that, if $X_i$ has c.d.f. $F_i(x_i)\,,$
then $F^{-1}_i(X_i)$ is uniformly distributed on $[0, 1]\,.$

The associated **copula** function is defined as the joint
c.d.f. of $U_1,...,U_n\,.$
If $F$ admits density $f\,,$ so does the copula, and

$f(x_1,...,x_n) = c(F_1(x_1),...,F_n(x_n)) f_1(x_1) \cdots f_n(x_n)\,.$

We can therefore split the construction of the joint probability
into the marginal distributions and the copula.

There are many possible choices for the copulas, and we will only discuss
the Student-t copulas.
Let us consider a multivariate standard Student-t distribution with $\nu$
degrees of freedom and correlation matrix $\Sigma\,.$
The associated copula can be constructed as

$$
C^t_{\nu, \Sigma}(u_1,...u_n) = t_{\nu, \Sigma}(t^{-1}_{\nu}(u_1),...,t^{-1}_{\nu}(u_n))\,.
$$

Also drawing random numbers is immediate by using copulas.
In order to sample from $C^t_{\nu, \Sigma}$ you can draw 

$$
X=(X_1,...,X_n) \sim t_{\nu, \Sigma}
$$

and then return 

$$U=(t^{-1}_\nu(Z_1),...,t^{-1}_\nu(Z_n)) \sim C^t_{\nu, \Sigma}\,.$$

We also have that

$$(F_1(U_1),...,F_n(U_n)) \sim F\,,$$

so we can also easily sample from $F\,.$