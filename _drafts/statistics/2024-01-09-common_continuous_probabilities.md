---
layout: post
title: "Common continuous probabilities"
categories: /statistics/
subgategory: Introduction
section: 4
tags: /discrete_probabilities/
date: "2024-01-09"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Derivation of some useful one-dimensional continuous probabilities"
---

In the previous post we derived some of the most relevant discrete probability
distributions. Here we will continue the discussion with some continuous
univariate distribution.

## Normal distribution

The normal distribution has been the most relevant continuous
distribution for centuries.
Consider a continuous distribution $f(x)$ with support $\mathbb{R}\,,$
and assume it is well peaked around $x=\mu\,.$
Since it is positive on the entire real axis, we can write

$$
f(x) = \exp{\log(f(x))} = \exp(g(x))
$$

Since we assume it is well peaked around $\mu$ we can expand $g(x)$
around $\mu\,,$ where we must have $g'(\mu) = 0$ and $g''(\mu)<0\,.$
We can therefore write

$$
f(x) \approx e^{g(\mu) + g'(\mu)(x-\mu) + \frac{1}{2}g''(\mu)(x-\mu)^2}
= e^{g(\mu)} e^{-\frac{(x-\mu)^2}{2 \sigma^2}}
$$

where $\sigma = \frac{1}{2\sqrt{-g''(\mu)}}\,.$

We define the normal probability distribution as

$$
p(x | \mu, \sigma) \propto e^{-\frac{(x-\mu)^2}{2 \sigma^2}}
$$

We must now normalize the above probability to one.

$$
\begin{align}
\int_{-\infty}^\infty dx e^{-\frac{(x-\mu)^2}{2 \sigma^2}} & =
\int_{-\infty}^\infty dx e^{-\frac{x^2}{2 \sigma^2}}  
\\ &
=
\int_{-\infty}^\infty dy e^{-\frac{y^2}{2}} \frac{d (y \sigma)}{dy}
\\ &
= \sigma \int_{-\infty}^\infty dy e^{-\frac{y^2}{2}} 
\end{align}
$$

In order to evaluate the above integral, we must go to two dimensions.
The method to evaluate the above integral is well known (see the [Wolfram site](https://mathworld.wolfram.com/GaussianIntegral.html) as an example), and it consists
to observe the following:

$$
\begin{align}
\int_{-\infty}^\infty dx e^{-\frac{x^2}{2}} 
= &
\left(
\int_{-\infty}^\infty dx e^{-\frac{x^2}{2}} 
\int_{-\infty}^\infty dy e^{-\frac{y^2}{2}} 
\right)^{\frac{1}{2}}
\\
&
= \left(\int_{\mathbb{R}^2} dx dy e^{-\frac{x^2+y^2}{2}}\right)^{\frac{1}{2}}
\\
&
=
\left(
\int_0^\infty r dr \int_0^{2 \pi} d\theta e^{-\frac{r^2}{2}}
\right)^{\frac{1}{2}}
\\
&
=
\left(
2 \pi \int_0^\infty r dr e^{-\frac{r^2}{2}}
\right)^{\frac{1}{2}}
\\
&
=
\left(
2 \pi \int_{0}^\infty  dt e^{-t}
\right)^{\frac{1}{2}}
\\
&
= \sqrt{2 \pi}
\end{align}
$$

We can therefore define

$$
p(x | \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

$$
\begin{align}
\mathbb{E}[X]
&=
\int_{-\infty}^\infty dx x
\frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\\
&
=
\int_{-\infty}^\infty dx (x-\mu)
\frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
+
\int_{-\infty}^\infty dx \mu
\frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\\
&
=\mu
\end{align}
$$

where we dropped the integral with the $(x-\mu)$ term, as it is odd,
so it vanishes.


$$
\begin{align}
Var[X] &= \frac{1}{\sqrt{2 \pi \sigma^2}} \int_{-\infty}^\infty dx (x-\mu)^2 e^{-\frac{(x-\mu)^2}{2 \sigma^2}} =
\\
& = \frac{1}{\sqrt{2 \pi \sigma^2}} \int_{-\infty}^\infty dx x^2 e^{-\frac{x^2}{2 \sigma^2}}  
\\
& = \frac{1}{\sqrt{2 \pi \sigma^2}} 2 \int_{0}^\infty dx x^2 e^{-\frac{x^2}{2 \sigma^2}}
\\
& = \frac{1}{\sqrt{2 \pi \sigma^2}} 2 \int_{0}^\infty dx x\left(xe^{-\frac{x^2}{2 \sigma^2}}\right)
\\
& = \frac{1}{\sqrt{2 \pi \sigma^2}} 2 \int_{0}^\infty dx x\frac{d}{dx}\left(-\sigma^2 e^{-\frac{x^2}{2 \sigma^2}}\right)
\\
& = -2\sigma^2 \frac{1}{\sqrt{2 \pi \sigma^2}} \int_{0}^\infty dx x\frac{d}{dx}\left( e^{-\frac{x^2}{2 \sigma^2}}\right)
\\
&
= -\left( \frac{1}{\sqrt{2 \pi \sigma^2}} 2 \sigma^2 x e^{-\frac{x^2}{2 \sigma^2}} \right)_0^\infty
+
2 \sigma^2 \frac{1}{\sqrt{2 \pi \sigma^2}} \int_0^\infty e^{-\frac{(x-\mu)^2}{2 \sigma^2}}
\\
&
= \sigma^2 \frac{1}{\sqrt{2 \pi \sigma^2}} \int_{-\infty}^\infty e^{-\frac{(x-\mu)^2}{2 \sigma^2}}
\\
&
= \sigma^2
\end{align}
$$

The initial procedure, that we used to define the Normal (also known as Gaussian)
distribution, is known as the saddle point (in physics) or [Laplace](https://en.wikipedia.org/wiki/Laplace%27s_approximation) (in statistics) approximation for $f\,,$
and it turns out to be vary useful in a variety of situations.

## Log-normal distribution

Consider a random variable $X$ with zero mean and unit variance, then
the random variable 

$$Z = e^{\mu + \sigma X}$$

follows by definition a **log-normal** distribution with parameters $\mu$ and $\sigma\.$

The inverse transformation is $x = \frac{\log{z}-\mu}{\sigma}\,,$
so $\frac{dx}{dz} = \frac{1}{z}$ therefore

$$
p(z | \mu, \sigma) = \frac{1}{z \sigma\sqrt{2 \pi}}e^{-\frac{(\log(z)-\mu)^2}{2\sigma^2}}
$$

## Chi-squared distribution

Given $d$ normal distributions $X_1,\dots, X_d\,,$ with $\mu=0$ and $\sigma=1\,,$
we define the **chi squared distribution with d degrees of freedom**
as
$$
X = \sum_{i=1}^d X_i^2
$$

In order to obtain the pdf of the $\chi^2$ 
we will roughly follow [this demostration](https://en.wikipedia.org/wiki/Proofs_related_to_chi-squared_distribution).
Consider an arbitrary expectation value involving only $X\,.$
By defining $R = \sqrt{X}\,,$ we have

$$
\mathbb{E}[f(X)]
=
\int \prod_{i=1}^d dx_i f(r^2) \frac{e^{-\frac{r^2}{2}}}{(2 \pi)^{\frac{d}{2}}}
$$

We can decompose 

$$
\prod_{i=1}^d dx_i = d\Omega r^{d-1} dr 
$$

and $$\int d\Omega = \frac{ 2 \pi^{d/2}}{\Gamma(d/2) }\,,$$
where we introduced the Euler gamma function

$$
\Gamma(z) = \int_0^\infty dx e^{-x} x^{z-1}
$$

We can now perform the transformation $x = \sqrt{r}\,,$
so $dr = \frac{2 dx}{\sqrt{x}}\,,$
and obtain

$$
\int \prod_{i=1}^d dx_i f(r^2) \frac{e^{-\frac{r^2}{2}}}{(2 \pi)^{\frac{d}{2}}}
= \frac{1}{2^{d/2} \Gamma(d/2)} \int_0^\infty dx e^{-\frac{x}{2}} x^{\frac{d}{2}-1} f(x)
= \int dx f(x) p(x | d)
$$

In this way we can identify the 

$$
p(x | d) = \frac{1}{2^{d/2}\Gamma(d/2)}e^{-\frac{x}{2}} x^{\frac{d}{2}-1} 
$$

An equivalent way to obtain 
$$p(x | d)$$
is to derive it from

$$
p(x | d) = 
\int \prod_{i=1}^d dx_i \delta(x-\sum_i x_i^2) \frac{e^{-\frac{\sum_i x_i^2}{2}}}{(2 \pi)^{\frac{d}{2}}}
$$

where $\delta(x-x_0)$ is the Dirac delta, defined by the relation
$$ \int dx f(x) \delta(x-x_0) = f(x_0)\,.$$

Since by definition
$$X = \sum_i X_i^2$$
and we assumed that $$\mathbb{E}[X_i^2] = 1\,,$$
we have that $$\mathbb{E}[X] = d\,.$$

Moreover, if $X$ follows $$\chi^2_k$$ and $Y$ follows $$\chi^2_s\,,$$
the variable $Z=X+Y$ follows $$\chi^2_{k+s}\,.$$

## Student's t distribution
Given a normally distributed variable $X$ with $\mu=0$ and $\sigma=1$
and given $Y$ distributed according to $\chi^2_d\,,$
we define the Student's t distribution with $d$ degrees of freedom as the one
which describes

$$
T = \frac{X}{\sqrt{Y/d}}
$$

By proceeding as before we have

$$
p(t | d) = \int_{-\infty}^\infty dx \int_0^\infty dy \frac{1}{\sqrt{2 \pi}}e^{-\frac{x^2}{2}} 
\frac{1}{2^{d/2} \Gamma(d/2)} e^{-\frac{y}{2}} y^{\frac{d}{2}-1}
\delta\left(t - \frac{x}{\sqrt{y/d}}\right)
$$

As shown in [these notes](https://shoichimidorikawa.github.io/Lec/ProbDistr/t-e.pdf), 
the above integral can be computed by defining $$z(x)=\frac{x}{\sqrt{y/d}}\,,$$
so that

$$
\begin{align}
p(t | d) & 
= \int_{-\infty}^\infty dz \int_0^\infty dy  \sqrt{y/d} \frac{1}{\sqrt{2 \pi}}e^{-\frac{z^2 y}{2d}} 
\frac{1}{2^{d/2} \Gamma(d/2)} e^{-\frac{y}{2}} y^{\frac{d}{2}-1}
\delta\left(t - z\right) \\
&
= \int_0^\infty dy  \sqrt{y/d} \frac{1}{\sqrt{2 \pi}}e^{-\frac{t^2 y}{2d}} 
\frac{1}{2^{d/2} \Gamma(d/2)} e^{-\frac{y}{2}} y^{\frac{d}{2}-1}
\\
&
= 
\frac{1}{2^{d/2} \sqrt{2 \pi d}\Gamma(d/2)} 
\int_0^\infty dy 
e^{-\frac{y}{2}(1+\frac{t^2}{d})} y^{\frac{d-1}{2}}
\end{align}
$$

If we now define $u = \frac{y}{2}(1+\frac{t^2}{d})$ we have

$$
\begin{align}
p(t | d) =
&
\frac{1}{\sqrt{\pi d}\Gamma(d/2)} \left(1+\frac{t^2}{d}\right)^{-\frac{d+1}{2}}
\int_0^\infty  du
e^{-u} u^{\frac{d-1}{2}}
\\
&
\frac{1}{\sqrt{\pi d}\Gamma(d/2)} \left(1+\frac{t^2}{d}\right)^{-\frac{d+1}{2}}
\Gamma\left(\frac{d+1}{2}\right)
\end{align}
$$

We can furthermore simplify by using the beta function

$$
B(x, y) = \int_0^1 dt\,  t^{x-1}(1-t)^{y-1} = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}
$$

so

$$
\frac{\Gamma\left(\frac{d+1}{2}\right)}{\Gamma\left(\frac{d}{2}\right)}
= \frac{\Gamma\left(\frac{1}{2}\right)} {B\left(\frac{d}{2}, \frac{1}{2}\right)}
= \frac{\sqrt{\pi}}{ B\left(\frac{d}{2}, \frac{1}{2}\right)}
$$

and write

$$
p(t | d) = \frac{1}{\sqrt{d} B\left(\frac{d}{2}, \frac{1}{2}\right)} \left(1+\frac{t^2}{d}\right)^{-\frac{(d+1)}{2}}
$$

Since the distribution is even with respect to 0, we have

$$
\mathbb{E}[X] = 0
$$

The calculation of the variance is rather complicated, and the detailed
proof can be found [here](https://proofwiki.org/wiki/Variance_of_Student%27s_t-Distribution).
We have that

$$
\begin{align}
&
\int_{-\infty}^\infty dx x^2\left(1+\frac{x^2}{d}\right)^{-\frac{(d+1)}{2}}
\\
&
=
d^{3/2}\int_{-\infty}^\infty d(x/\sqrt{d}) \frac{x^2}{d}\left(1+\frac{x^2}{d}\right)^{-\frac{(d+1)}{2}}
\\
&
=d^{3/2} \int_{-\infty}^\infty dy y^2\left(1+y^2\right)^{-\frac{(d+1)}{2}}
\\
&
=2d^{3/2} \int_{0}^\infty dy y^2\left(1+y^2\right)^{-\frac{(d+1)}{2}}
\end{align}
$$

Let us define $$u = y^2 (1+y^2)^{-1}\,,$$ we can rewrite the last integral as

$$
\begin{align}
&
\int_{-\infty}^\infty dx x^2\left(1+\frac{x^2}{d}\right)^{-\frac{(d+1)}{2}}
\\
&
=
d^{3/2} \int_{0}^1 du u^{\frac{1}{2}} (1-u)^{\frac{d-4}{2}}
\\
&
= d^{3/2} B\left(\frac{3}{2}, \frac{d-2}{2}\right)
\end{align}
$$

Now observe that

$$
\begin{align}
d \frac{ B\left(\frac{3}{2}, \frac{d-2}{2}\right) }{B\left(\frac{d}{2},\frac{1}{2}\right)}
&
=
d \frac{\Gamma(3/2)\Gamma((d-2)/2) \Gamma((d+1)/2)}{\Gamma(d/2)\Gamma(1/2)\Gamma((d+1)/2)}
\\
&
=
d \frac{1/2}{(d-2)/2} = \frac{d}{d-2}
\end{align}
$$

where we used $\Gamma(z+1) = z \Gamma(z)\,.$
The above integral only exists if $d>2\,,$ otherwise the beta function
develops a pole.

If $d=1$ the distribution is also known as Cauchy distribution.

In the limit $d \rightarrow \infty$ we $(1+\frac{x}{n})^n \approx e^{x}\,,$ so

$$
p(t | d) \propto \left(1+\frac{t^2}{d}\right)^{-\frac{(d+1)}{2}}
= \left(\left(1+\frac{t^2}{d}\right)^{d+1}\right)^{1/2} 
\approx \left(\left(1+\frac{t^2}{d}\right)^{d}\right)^{1/2}  \approx e^{-\frac{t^2}{2}}\,,
$$

The normalization factor can be approximated as

$$
\sqrt{d}B\left(d/2, 1/2\right) = \sqrt{d} \frac{\Gamma\left(d/2\right)
\Gamma\left(1/2\right)}{\Gamma\left(d/2+1/2\right)}
\approx \sqrt{d} \frac{\Gamma\left(d/2\right)\Gamma\left(1/2\right)}{
\Gamma\left(d/2\right) \left(d/2\right)^{1/2}} = \sqrt{2\pi}
$$

so in the large $d$ limit the Student's t distribution can
be well approximated by a Gaussian distribution.

## Exponential distribution

Here we will derive the exponential distribution by a Poisson process,
proceeding as theorem 35.1 [Dennis Sun](https://dlsun.github.io/probability/exponential.html).
Consider a Poisson process with mean $\mu = \lambda t\,,$
where $t>0\,.$
Consider the random variable $T$ representing the first event.
The probability to wait a time $t$ before $T$ is $$P(T<t) = 1-P(t>T)\,.$$
Since the process is Poissonian, $$P(t>T)$$ can be computed as $$p(k=0 | \lambda t)\,,$$
so

$$
P(T<t | \lambda) = 1- e^{-\lambda t} \frac{(\lambda t)^0}{0!} = 1-e^{-\lambda t}
$$

The corresponding pdf is

$$
p(t | \lambda) = \frac{d}{dt}(1-e^{-\lambda t}) = \lambda e^{-\lambda t}
$$

and this is defined as the exponential distribution.
We have that

$$
\begin{align}
\mathbb{E}[T] & = \int_0^\infty dt \lambda t e^{-\lambda t}
\\
&
= \frac{1}{\lambda} \int_0^\infty dx x e^{-x}
\\
&
= -\frac{1}{\lambda} \int_0^\infty dx x \frac{d}{dx}e^{-x}
\\
&
= -\frac{1}{\lambda} \left(x \frac{d}{dx}e^{-x}\right)_0^\infty
+\frac{1}{\lambda} \int_0^\infty e^{-x}
\\
&
= \frac{1}{\lambda}
\end{align}
$$

In the same fashion we can show

$$
Var[T] = \frac{1}{\lambda^2}
$$

## Gamma distribution

A very useful generalization of the exponential distribution is
given by the gamma distribution, with pdf defined by

$$
p(x | \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1}e^{-\beta x}
$$

The mean for the Gamma distribution reads

$$
\begin{align}
\mathbb{E}[X] &
= \int_0^\infty dx x \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1}e^{-\beta x}
\\
&
=\frac{\beta^\alpha}{\Gamma(\alpha)} \frac{1}{\beta^{\alpha+1}}\int_0^\infty dy  y^{\alpha}e^{-y}
\\
&
=\frac{\Gamma(\alpha+1)}{\beta \Gamma(\alpha)} 
\\
&
=\frac{\alpha}{\beta}
\end{align}
$$

Analogously we have that

$$
Var[X] = \frac{\alpha}{\beta^2}
$$

Notice that both the $\chi^2$ distribution and the exponential distribution
are special cases of the gamma distribution.

## Uniform distribution

The uniform distribution on an interval $[a, b]$ is defined as

$$
p(x | a, b) = \frac{1}{b-a}
$$

We immediately have

$$
\mathbb{E}[X] = \frac{1}{b-a} \int_a^b dx x = \frac{1}{b-a}\left(\frac{b^2-a^2}{2}\right)
= \frac{b+a}{2}
$$

Analogously

$$
Var[X] = \mathbb{E}\left[\left(X-\frac{b+a}{2}\right)^2\right] = \frac{1}{b-a} \int_a^b dx \left(x-\frac{a+b}{2}\right)^2
= (b-a)^2 \int_0^1 dy \left(y-\frac{1}{2}\right)^2 = \frac{(b-a)^2}{12}
$$




## Beta distribution

Another useful distribution is the beta distribution, defined on $[0, 1]$
from the pdf

$$
p(x | \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
$$

This distribution has expectation value

$$
\mathbb{E}[X] = \frac{B(\alpha+1, \beta)}{B(\alpha, \beta)}
= \frac{\Gamma(\alpha+1)\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\alpha+\beta+1)}
= \frac{\alpha}{\alpha + \beta}
$$

In the same way we can show that

$$
Var[X] = \frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
$$

## Location scale families

Some of the pdf we defined above do not have any free parameter. We can however easily promote them to a location-scale family of distributions.

If $f(x)$ is a pdf, then 
$$p(x | \mu, \sigma)=\frac{1}{\sigma} f\left(\frac{x-\mu}{\sigma}\right)$$ is a pdf too for $\mu \in \mathbb{R}$ and $\sigma>0$, and it defines a 
**location-scale parameter family** of distributions.

## Conclusions

We discussed some of the most common continuous distribution.
In the next post we will finally start discussing about how to
infer properties of the distribution from the data.
