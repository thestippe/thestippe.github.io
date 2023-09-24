---
layout: post
title: "Causal inference: a general introduction"
categories: course/various/
tags: /causal-intro/
---

## Causality as counterfactual evidence

In the last years causal inference gained a lot of attention, both from academia and outside from it.
Of course, you heard that *association does not imply causation*, as you will find an association
between ice-cream consumption and the number of deaths by drowing.
However, outside from the classes for statisticians, the very important question *when does association imply causation*
is rarely discussed.
We should first try and clarify what do we mean by causation, as the popular concept of causation
is too vague for any rigorous discussion.
This topic has been debated is philosophy since centuries, and if you are interested about this aspect
I found [this](https://iep.utm.edu/causation/) introductory article very helpful.

We won't dig into the philosophical debate, and simply use the **counterfactual** approach:
we say that an action $T$, which is usually called treatment or intervention,
causes $Y$ if, when $T$ disappears, then $Y$ disappears.
Our definition implies that we must switch off $T$ so,
in order for our definition to be meaningful, we must be able *at least hypothetically*
to manipulate $T$, and in this case we say that $T$ is **manipulable**. 

A meaningful question is if a medicine cures the illness, as we may or may not take the medicine,
but we cannot ask whether age causes heart attack, as we can hardly imagine to change one person's age.
The concept of manipulability depends on the context, as we may ask if increasing age causes a reduction
in the chances to be considered for a certain working position. In this case we may manipulate the age by simply
changing it on the CV and check if the company calls for a job interview.

The counterfactual definition is not precise enough,
as it may happen that $Y$ appears only when both $T$ and $Z$ appears, so in this
case an obvious question is whether $T$ or $Z$ is the cause of $Y$.
We will always assume that we want to investigate only one cause at time, so either we want to determine
if, given $T$, then $Z$ is a cause of $Y$ or vice versa if, given $Z$, then $T$ causes $Y$.
This imply that, when we investigate causality, we must change the hypothetical cause
by keeping everything else unchanged.

A very common source of confusion is the question causal inference tries and answer:
as explained in Gelman's preprint [Causality and Statistical Learning](https://arxiv.org/pdf/1003.2619.pdf)
when talking about causality, there are two main questions one could ask:
- backward causal inference: what are the causes of a given effect?
- forward causal inference: what are the effects of a given cause?

While there are many accepted methods to investigate the forward causal inference,
backward causal inference is a slippery terrain, as one could also 
say that the cause of the cause is the cause.
In fact, it is not uncommon to have that $A$ causes $B$, $B$ causes $C$ and $C$ causes $D$,
and in this case we will consider both $A$, $B$ and $C$ as causes of $D$.

## The fundamental problem of causal inference

Let us assume for now that $T$ is a binary quantity with values $0$ and $1$, and let us indicate
the value of $Y$ when $T=0$ as $y_0$ while $y_1$ is the value of $Y$ whet $T=1$.
In order to assess whether $T$ causes $Y$ we must compare $y_1$ with $y_0$.

The exact way we want to compare these two quantities depends on the context.
Most of the time what one wants to quantify is the so called effect, defined as $\delta = y_1-y_0$, but in some cases one may prefer to
obtain informations about the relative risk $y_1/y_0$.
In any case, what we want to do is to compare both quantities and verify if they differ.

In most textbooks one defines the function

$$Y(\tau) = \tau y_1 + (1-\tau) y_0$$

so

$$\delta = Y(1) - Y(0)$$

One generally refers to the quantities $Y(0)$ and $Y(1)$ as the potential outcomes.
More precisely, the potential outcome is represented by the previous quantities before the experiment,
while during the experiment one measures the observed outcome, and the remaining quantity is the counterfactual outcome.
Since we assume that these quantities are the same, we will always refer to the potential outcomes.

As we previously stated, $y_1$ and $y_0$ represent $Y$ when $T=1$ or $0$ respectively, but everything else is
unchanged. This makes always impossible to measure the causal effect, as we cannot simultaneously realize $T=0$
and $T=1$ by keeping everything else, included the moment and the individual, unchanged,
and this is called the **fundamental problem of causal inference**.

In order to better understand why this is a problem, let us assume that we have a population,
and that we somehow split the population into two subpopulation.
The first subpopulation is then treated, so they are assigned to the $T=1$ group,
while the second one is not, and for them $T=0$.
We then take the average on each subpopulation what we are estimating is
$\mathbb{E}[Y | T=1]$ and $\mathbb{E}[Y | T=0]$ respectively.
On the other hand, what we really want to quantify is

$$\mathbb{E}[\delta] = \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]$$

Let us indicate with $y^i_\tau$ the observed outcome on the individual $i$ of the population
when this undergoes to treatment $\tau$.
We generally have that the outcome will both on the treatment and on some set of
relevant covariate (auxiliary quantities) $X$, that we assumed we measured for each individual.

| i | T |    Y  |   Y(0)  |   Y(1)  |  X  | Y(1) - Y(0) |
|---|---|-------|---------|---------|-----|-------------|
| 1 | 0 | $y^1_0$ |   $y^1_0$ | $y^1_1=?$ |$x^1$|  $y^1_1-y^1_0=?$  |
| 1 | 0 | $y^2_0$ |   $y^2_0$ | $y^2_1=?$ |$x^2$|  $y^2_1-y^2_0=?$  |
| 1 | 0 | $y^3_0$ |   $y^3_0$ | $y^3_1=?$ |$x^3$|  $y^3_1-y^3_0=?$  |
| 4 | 1 | $y^4_1$ | $y^4_0=?$ |  $y^4_1$  |$x^4$|  $y^4_1-y^4_0=?$  |
| 4 | 1 | $y^5_1$ | $y^5_0=?$ |  $y^5_1$  |$x^5$|  $y^5_1-y^5_0=?$  |
| 4 | 1 | $y^6_1$ | $y^6_0=?$ |  $y^6_1$  |$x^6$|  $y^6_1-y^6_0=?$  |

We have that

$$
\begin{aligned}
&
\mathbb{E}[Y | T=0] = \frac{y^1_0 + y^2_0 + y^3_0}{3}
\\
&
\mathbb{E}[Y | T=1] = \frac{y^4_1 + y^5_1 + y^6_1}{3}
\end{aligned}
$$

on the other hand

$$
\begin{aligned}
&
\mathbb{E}[Y(0)] = \frac{y^1_0 + y^2_0 + y^3_0 + y^4_0 + y^5_0 + y^6_0}{6}
\\
&
\mathbb{E}[Y(1)] = \frac{y^1_1 + y^2_1 + y^3_1 + y^4_1 + y^5_1 + y^6_1}{6}
\end{aligned}
$$

We cannot measure the quantities marked with the question mark,
so the fundamental problem of causal inference is a missing value problem.
The different terms entering into the two expressions don't allow us
to simply substitute the associational quantities with the causal ones,
and this is why association is not causation.

As we will show briefly, however, when a set of rather stringent condition
holds, we are allowed to replace the causal quantities with the associational ones.
However, there is no way to verify if these conditions are met, and one should
rely on some external source of knowledge about the population
in order to assess at which approximation level these conditions hold.

The stronger condition that might hold is **ignorability**, also called **exchangeability**

$$ Y(0), Y(1) \perp\!\!\!\!\perp T $$

We are thus assuming that the probability of being treated is independent on the outcome.
This is of course a very strong assumption, and the fact that in most observational
studies this condition is not met implies a wrong estimation of the effect.
As an example, if we are performing an observational study on a medicine, usually only people which
are sick and so will benefit by the medicine, will take the medicine and, so, will be included in the
treated group, while in the untreated group we may have sick people as well as healthy people.
So we must introduce a confounder $x=sick, healthy$ to account for this.
The ignorability assumption states that we must be allowed to exchange the two groups without affecting the outcome.

Under this condition we have that

$$
\mathbb{E}[Y | T=1] - \mathbb{E}[Y | T=0] = \mathbb{E}[Y(1) | T=1] - \mathbb{E}[Y(0) | T=0] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)] = \mathbb{E}[Y(1)-Y(0)] = \mathbb{E}[\delta]
$$

In the previous equation we used 

$$
\mathbb{E}[Y | T=t] = \mathbb{E}[Y(t) | T=t]
$$

This only holds if we assume that, if $T=t$, then $Y = Y(T=t)$.
This assumption goes under the name on **consistency**, and it is not only a mathematical
requirement, but an operational one.
Consistency requires that the treatment must be well specified:
the treatment must not be "get some medicine" but should rather be "take 15 mg of medicine every 8 hours for 7 days".

A slightly weaker condition with respect to ignorability is **conditional ignorability** or **unconfoundedness**

$$ Y(0), Y(1) \perp\!\!\!\!\perp T | X $$

So given the confounders $X$, the treatment probability $T$ is independent on the
potential outcome.

Let us assume we want to quantify the blood pressure reduction of a medicine.
It is more likely that people with a high blood pressure will take it.
We furthermore assume that the effect is higher on people with a high blood pressure.

Since it is more likely that people with high blood pressure will take the treatment,
ignorability doesn't generally hold in observational studies.
We can randomly sample from the population to overcome this, but it will be very hard to obtain
a representative sample of the population.
An easier way is to stratify by initial blood pressure,
and for each stratum randomly assign with a given probability
to the treatment group or to the test group, and in this way we are fulfilling conditional
ignorability.

Thus, unconfoundedness states that we are "controlling for" all the relevant quantities which
may affect the outcome, except the treatment.
This may only approximately hold: if the outcome depends on some genetic aspect of the individual
which is more common in a particular ethnic group, controlling for ethnicity would partially fulfill 
unconfoundedness.

When we assign the population we must be sure that, for each stratum,
both groups have at least one individual:

$$ 0 < P(T=t | X) < 1 \, \forall t$$

The previous hypothesis is named the positivity assumption.
Positivity implies that we can compare the treatment effect with the control for each value of the
covariates, since
for each subgroup we both have units which receive the treatment and units which
does not receive it.

If conditional ignorability holds:

$$
\begin{aligned}
 \mathbb{E}[Y(1)-Y(0)|X] 
 & = \mathbb{E}[Y(1)|X] - \mathbb{E}[Y(0)|X] \\
 & = \mathbb{E}[Y(1)| T=1, X] - \mathbb{E}[Y(0)|T=0, X] \\
 & = \mathbb{E}[Y| T=1, X] - \mathbb{E}[Y|T=0, X] \\
\end{aligned}
$$

By taking the average over $X$

$$
\mathbb{E}[\delta] = \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}_X[ \mathbb{E}[Y(1) - Y(0) | X] ] 
 = \mathbb{E}_X[ \mathbb{E}[Y |T=1, X] ] - \mathbb{E}_X[ \mathbb{E}[Y |T=0, X] ]
$$

The equality between the first and the last term of this equation is called the **adjustment formula**.

Let us now write explicitly the adjustment formula for $X$ discrete:

$$
\begin{align}
&
\mathbb{E}_X[ \mathbb{E}[Y |T=1, X] ] - \mathbb{E}_X[ \mathbb{E}[Y |T=0, X] ]  
\\
& =  \sum_{x}P(X=x) \sum_{y} y \left(P(Y=y|T=1, X=x) - P(Y=y| T=0, X=x) \right) \\
& =   \sum_{x}P(X=x) \sum_{y} y \left(\frac{P(Y=y,T=1, X=x)}{P(T=1, X=x)} - \frac{P(Y=y,T=0, X=x)}{P(T=0, X=x)}\right) \\
= & \sum_{x}P(X=x) \sum_{y} y \left(\frac{P(Y=y,T=1, X=x)}{P(T=1| X=x) P(X=x)} - \frac{P(Y=y,T=0, X=x)}{P(T=0| X=x) P(X=x)}\right) 
\\
& = 
\sum_{x}\sum_{y} y \left(\frac{P(Y=y,T=1, X=x)}{P(T=1| X=x)} - \frac{P(Y=y,T=0, X=x)}{P(T=0| X=x)}\right) 
\end{align}
$$

The first equivalence comes from the definition of conditional probability,
the second one from the hypothesis that $P(T, X) = P(T | X) P(X) $ so that $T$ causally depends on $X\,.$
You should notice that the denominators are finite thanks to the positivity hypothesis.

There is one more hypothesis that we have hidden into our discussion:
we have been assuming all the time that the outcome of the i-th unit only depend on the i-th treatment unit,
and does not depends on the other treatment's unit.
This requirement is of course not always satisfied, and it's called the **no interference** assumptions:

$$ Y_i(t_1, t_2, ..., t_{i-1}, t_i, t_{i+1}, ..., t_n) = Y_i(t_i) $$

So each individual's outcome only depends on his own treatment and not on the treatment of other individuals.
This implies that, if we are checking the effect of a product in some tomato field, we must be sure that the product does not goes in another studied field by mistake.
Another case can be a study where we are studying an experimental study program in a class.
If a student is selected in the treatment group and a friend of his is not, the latter could be sad for not being selected and his outcome could be lowered.
Generally, a good strategy to enforce this requirement is to take well separated units and isolating each unit from the other units during the experiment.

## Conclusion and take home message

As we can see, under some strict assumptions we can perform causal inference in observational studies as well as in randomized studies.
However, quoting Cochran:

***observational studies are are interesting and challenging field which demands a good deal of humility, since our claim are groping toward the truth.***



## Additional readings

[Hernàn, Robins; **Causal inference, what if**, Chapman & Hall/CRC (2020)](https://www.hsph.harvard.edu/wp-content/uploads/sites/1268/2022/11/hernanrobins_WhatIf_13nov22.pdf)
