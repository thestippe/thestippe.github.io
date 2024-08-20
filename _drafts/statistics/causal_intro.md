---
layout: post
title: "Causal inference"
categories: /statistics/
subcategory: "Causal inference"
tags: /causal_intro/
date: "2024-06-09"
# image: "/docs/assets/images/perception/eye.jpg"
description: "When association implies causation"
section: 0
---

In this post we will try and clarify when it is possible to make statements
about causation rather than sticking to statistical association,
and we will do so on the basis of Rubin's potential outcomes.

The main reference for this part will be the material in
[these](https://www.bradyneal.com/Introduction_to_Causal_Inference-Dec17_2020-Neal.pdf)
notes by Brady Neal, but I strongly recommend to read the textbook by Guido Imbens
(who, in 2021, shared the Nobel prize for economics with Joshua Angrist and David Card for their works on causal inference) and Carl Rubin
(who first developed the potential outcomes framework).

## The counterfactual definition of causality

You may have heard the mantra "association is not causation" or the more colloquial 
(but less accurate) "correlation is not causation".
Correlation is a statistical measure linear dependence,
while association generally means statistical dependence.
However the exact meaning of causation is never given,
and the first part of these notes will be devoted to clarify what do we mean with causation.

Let us assume that we took a medicine because we had a headache,
what do we mean when we say that the medicine
caused the headache to disappear? It means that, if we hadn't taken the medicine,
the headache wouldn't have gone away.

We will therefore stuck to the counterfactual definition of causation,
so we will say that an event causes an outcome if,
by removing the event then the outcome disappears.
The above definition works for binary outcomes, but has some
problems when we want to investigate causes which can take any real value.
More generally, we can say that an event causes an outcome if, by modifying
the cause, the outcome changes.
This definition already puts a strong constrain on what we can investigate,
since it requires that we must be able, at least in principle, to modify the
cause.

<div class="emphbox">
There's no causation without manipulation.
</div>

While the meaning of the above sentence may seem obvious at a first sight,
you should carefully think about it when making causal inference.
If you want to assess the effect of the ethnicity on the probability of being hired,
you may not be able to manipulate someone's ethnicity,
but you can still manipulate people's perception of ethnicity by modifying 
the CV. 

When talking about causality, one can be either interested in the determination
of the effect of a cause (e.g. does my headache disappears when I take medicine?)
or the cause of an effect (e.g. is my headache gone because I took the medicine?).

Within the counterfactual framework, and in science in general, one generally wants
to assess the effect of a specific cause.
Determining the cause of an effect, in fact, is an ill-posed question, as one could
find a cause of the cause, a cause of the cause of the cause and so on.

A relevant aspect which we must keep in mind is that there could be more than one
cause. We know that, in order to light a fire, we need oxygen, heat and fuel,
and all the above are necessary conditions for fire.

Let's assume that we want to assess if heat causes fire ignition,
and we perform an experiment to determine it.
If we first provide both three the elements,
and we then remove oxygen and heat, we can't conclude anything about the
causal relation between heat and fire, since we also removed the oxygen.
The counterfactual definition of causality requires that
only the cause must change, while all the other elements must be unchanged.

## Potential outcomes

But how can we measure the effect of an event? Let us indicate with $T=1$
the case where the event happens, as an example we take a therapy,
while $T=0$ means that we do not take the therapy.
Suppose that the outcome of the event $T=0$ is $y_0$ while the outcome of $T=1$ is
$y_1\,.$ We define 

$$Y(t) = t y_1 + (1-t) y_0\,.$$

From a counterfactual point of view, a natural way to assess the causal
effect is via the **Individual Treatment Effect** ITE

$$
\tau = Y(1)-Y(0)
$$

The definition of $\tau$ is of course arbitrary, but quite general.
As an example, one could prefer taking
the ratio of the two, but then taking the logarithm we recover the above definition.

Despite on the exact definition, $\tau$ of course cannot be measured, as either we take the treatment and
we observe $Y(1)$ or we don't and we observe $Y(0)\,,$
and any reasonable definition of $\tau$ involves both the quantities.
This implies that, 

<div class=emphbox>
while our definition of effect may be individual,
its quantification can only be done on a larger sample.
</div>

We must therefore do an experiment in order to estimate it:
we collect $2N$ individuals and divide them into 2 groups.
Half individuals are treated with $T=1$ (the treatment group)
and half of them with $T=0$ (the control group).

In order to proceed, we will stick to the definition $\tau = Y(1)-Y(0)\,.$
The simplest estimate that we may do is the **Average Treatment Effect** ATE

$$ ATE = \mathbb{E}[Y_i(1) - Y_i(0)] =  \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]$$

where the average is meant both on the individual and on any other possible source
of randomness.

To clarify what we are doing, we can put the collected data as

| i | T |    Y  |   Y(0)  |   Y(1)  |  X  | Y(1) - Y(0) |
|---|---|-------|---------|---------|-----|-------------|
| 1 | 0 | $y^1$ |   $y^1$ |    ?    |$x^1$|     ?       |
| 2 | 0 | $y^2$ |   $y^2$ |    ?    |$x^2$|     ?       |
| 3 | 0 | $y^3$ |   $y^3$ |    ?    |$x^3$|     ?       |
| 4 | 1 | $y^4$ |    ?    |  $y^4$  |$x^4$|     ?       |
| 5 | 1 | $y^5$ |    ?    |  $y^5$  |$x^5$|     ?       |
| 6 | 1 | $y^6$ |    ?    |  $y^6$  |$x^6$|     ?       |

where $X$ represents other possibly relevant quantity where we must take
into account to estimate the averages or, in other words, any quantity we may suspect could
affect the outcome.
The estimate of the ATE is the so-called **fundamental problem of causal inference**,
and since the question marks can be seen as missing values,

<div class='emphbox'>
the fundamental problem of causal inference is a missing value problem. 
</div>

We did a step further, but still we don't know how to compute that quantity.

In writing the above table, we implicitly made the **no interference** assumptions, namely that

$$ Y_i(t_1, t_2, ..., t_{i-1}, t_i, t_{i+1}, ..., t_n) = Y_i(t_i) $$

So each unit's outcome only depends on his own treatment and not on the treatment of other individuals.
This implies that, if we are checking the effect of a product in some tomato field,
we must be sure that the product does not goes in another studied field by mistake.
Another case can be a study where we are studying an experimental study program in a class.
If a student is selected in the treatment group and a friend of his is not,
the latter could be sad for not being selected and his outcome could be lowered.

Generally, a good strategy to enforce this requirement is to take well separated units
and not letting them communicate during the experiment.

Notice that this is not a necessary requirement, but it greatly simplifies the discussion,
as it allows us to threat each unit independently on the others.

A quantity that is closely related to the ATE is the **associational difference**

$$ \mathbb{E}[Y|T=1] - \mathbb{E}[Y|T=0]$$

When are we allowed to replace the ATE with the associational difference?
In other words, when are we allowed to compute the average only over the observed 
values and replace the question marks with the appropriate average?

The assumptions that the observed data do not depend on the missing ones
is called **ignorability**, and it is one of the most important assumptions
in causal inference.
Ignorability can be written in mathematical language as

$$ Y(0), Y(1) \perp\!\!\!\!\perp T$$

where $$ A \perp\!\!\!\!\perp B$$ means that $A$ and $B$ are independent one on
the other one.


If ignorability holds, we are allowed to estimate the average of $Y(0)$
by only using the $T=0$ group and replace it in the $T=1$ group and vice versa,
and this is why this assumption is often named **exchangeability**.

We can mathematically write the exchangeability assumption as

$$
\mathbb{E}[Y(0) | T=0] = \mathbb{E}[Y(0) | T=1] = \mathbb{E}[Y(0)] 
$$

and

$$
\mathbb{E}[Y(1) | T=0] = \mathbb{E}[Y(1) | T=1] = \mathbb{E}[Y(1)] 
$$

The above assumption is almost equivalent to **identifiability** assumption:
a causal quantity $\mathbb{E}[Y(t)]$ is identifiable if it can be computed from a pure statistical quantity $\mathbb{E}[Y | T=t]$.

There are cases where exchangeability does not hold.
As an example, assume that you are testing a medicine, and that this medicine
is more effective on patients with a severe version of the disease you are treating.
If in the $T=1$ group we have
people with a more severe version of the disease than in the $T=0$
group we may not be allowed to exchange the two groups,
as we have no guarantee that the result would be invariant under the group exchange.

Let us decompose the associational difference as

$$
\mathbb{E}[Y(1) | T=1] - \mathbb{E}[Y(0) | T=0]
=
(\mathbb{E}[Y(1) | T=1] - \mathbb{E}[Y(0) | T=1])
+(\mathbb{E}[Y(0) | T=1] - \mathbb{E}[Y(0) | T=0])
$$

The associational difference can be decomposed as
the average treatment effect on the treated (the first parenthesis)
plus the sampling bias (the second parenthesis).

Consider the case where $Y$ is the health status of a person and the treatment is
the hospitalization.
The associational difference is simply the difference between the health status
of hospitalized patients and the health status of non-hospitalized people.
This is simply the sum between the effect of the hospitalization on hospitalized
patients plus the baseline health difference between hospitalized and non-hospitalized individuals.
In general, even if hospitalization improves health, the health status of those who go to
the hospital is generally worst than the other individuals.
Therefore, in absence of randomization, if we simply use the associational difference to assess the effect of
hospitalization, we may end up with the conclusion that health get worst due to hospitalization
simply because only sick people goes to the hospital.

If exchangeability does not hold, then $$\mathbb{E}[Y \vert T=0]$$ is different from
$$\mathbb{E}[Y \vert T=1]\,,$$ therefore the associational quantity $$
\mathbb{E}[Y \vert T=t]$$ is a biased estimator for $$\mathbb{E}[Y(t)]\,.$$

One possible way to ensure that exchangeability holds is to ensure that the missing
terms are randomly distributed.
This can be experimentally done by randomly assigning the
treatment $T$ to each unit, and in this case we are dealing with a randomized experiment.

In a randomized experiment, the treatment assignment does not depend on anything other
other than the result of a coin toss, therefore

$$
\mathbb{E}[Y(1)]-\mathbb{E}[Y(0)] = \mathbb{E}[Y(1)|T=1]-\mathbb{E}[Y(0)|T=0] = \mathbb{E}[Y | T=1]-\mathbb{E}[Y | T=0]
$$

We stress that this is only a statistical property, and it doesn't guarantee that the outcome
estimate of an experiment will be correct. 

In other words, as explained in [the breaktrhough 1974 Rubin paper](http://www.fsb.muohio.edu/lij14/420_paper_Rubin74.pdf):
<blockquote cite="https://hedibert.org/wp-content/uploads/2015/10/causality-meeting2.pdf">
Whether treatments are randomly assigned or not, no matter how carefully
matched the trials, and no matter how large N, a skeptical observer could always
eventually find some variable that systematically differs in the E (T=1) trials and C (T=0) trials.
<br>
Within the experiment there can be no refutation of this claim; only a logical
argument explaining that the variable cannot causally affect the dependent
variable or additional data outside the study can be used to counter it.
</blockquote>

Generally exchangeability is an unrealistic assumption, as it would impossible to verify
that $X$ and $Y$ are equally distributed with respect to all the 
relevant variables except for the treatment.
A weaker assumption is that the assigned treatment only depends on some
relevant quantity $X$ while the two groups are exchangeable
with respect to any other quantity.
This condition is called **conditional exchangeability** or **unconfoundedness** and it is indicated as

$$ Y(0), Y(1) \perp\!\!\!\!\perp T | X$$

If conditional exchangeability holds, we have that

$$
\begin{align}
 \mathbb{E}[Y(1)-Y(0)|X] 
 & = \mathbb{E}[Y(1)|X] - \mathbb{E}[Y(0)|X] \\
 & = \mathbb{E}[Y(1)| T=1, X] - \mathbb{E}[Y(0)|T=0, X] \\
 & = \mathbb{E}[Y| T=1, X] - \mathbb{E}[Y|T=0, X] \\
 \end{align}
 $$

 In order to get the ATE we must simply take the expectation value over $X$

 $$ \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}_X[ \mathbb{E}[Y(1) - Y(0) | X] ] 
 = \mathbb{E}_X[ \mathbb{E}[Y |T=1, X] ] - \mathbb{E}_X[ \mathbb{E}[Y |T=0, X] ] 
 $$

 And the equality between the first and the last term of this equation is called the **adjustment formula**.

In the above equation we assumed **consistency**, which can be written as

$$ T=t \Longrightarrow Y(T) = Y(t) $$

This means that the treatment must be well specified: the treatment must not be "get some medicine" but should rather be "take 15 mg of medicine every 8 hours for 7 days".
Only thanks to this hypothesis we can replace 

$$ \mathbb{E}[Y(T=t) | T=t] = \mathbb{E}[Y | T=t] \,.$$

This is not a necessary requirement, but it greatly simplifies the discussion, otherwise we would be forced to model
this random aspect too.
Notice that the concept of consistency is not a mathematical requirement, but rather a conceptual one,
and only agreement among domain experts can assess whether it holds or not.

In the literature it is often required the **Stable Unit Treatment Value Assumption** SUTVA, which is simply requiring consistency and no interference.

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


where the first equivalence comes from the definition of conditional probability,
the second one from the hypothesis that $$P(T, X) = P(T | X) P(X) $$
so that $T$ causally depends on $X\,.$

In order for this quantity to be finite we must require that both the denominators are
strictly positive, and since $P(T=0 \vert X) = 1 - P(T=1\vert X)$ we can write this requirement,
named the **positivity** assumption, as

$$ 0 < P(T=t | X) < 1 \, \forall t$$

In other words, for each value of X we must have some representative individual in
each group.
If, for some group, everyone receives the treatment or everyone receives the control,
we are not able to estimate the treatment versus control effect.

<!--
We can better see why randomization is important from the adjustment formula.
If the treatment assignment is not random, one cannot assume that $P(X)$ is the same for the two
groups, and therefore one should replace it with $P(X=x | T=t)\,.$
A careful observer could therefore always find some confounder which could differ
between the two groups, and claim that the difference in the effect is due to the
different distribution of that confounder.
Within randomization, this cannot happen, as the probability distribution
of $X$ is independent on $T\,.$
-->

## Some remarks

Our discussion on causality both applies to frequentist statistics
and to bayesian one. However, as pointed out by Rubin himself in his 1990
article "Formal mode of statistical inference for causal effects",
it is straightforward to apply fully bayesian methods to causal inference.
However, it is very easy to misuse it, as 

<blockquote cite="https://www.sciencedirect.com/science/article/abs/pii/0378375890900778">
there appears to be no formal requirement to make sure
that the models conform at all to reality. 
In practice, careful model monitoring is
needed, and for this purpose, the randomization-based approaches we have presented
can be regarded as providing useful guidelines.
</blockquote>

It is therefore crucial both to ensure that experimental setup fulfills the
above mentioned assumptions and that the statistical model is appropriate
in describing it.

## Conclusions

We introduced the counterfactual definition of causality, and we introduced
Rubin's potential outcomes. We also discussed under which conditions
we can compute the average treatment effect.


## Suggested readings

- <cite>Imbens, G. W., Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction. US: Cambridge University Press.<cite>
- <cite><a href='https://arxiv.org/pdf/2206.15460.pdf'>Li, Ding, Mealli (2022). Bayesian Causal Inference: A Critical Review</a></cite>
- <cite>Ding, P. (2024). A First Course in Causal Inference. CRC Press.</cite>

[^1]: But pay attention not to think that randomized tests are perfect. See [this article](https://www.bmj.com/content/363/bmj.k5094) as an example.
