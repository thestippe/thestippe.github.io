---
layout: post
title: "What is statistics"
categories: /statistics/
tags: /statistics
image: "/docs/assets/images/statistics/statistics.webp"
description: "What is statistics and how to use it."
---

Sometimes dataviz is not enough to get the desired information from your data.
This may happen for many reasons, either because you want to ask some hypothetical question *what would happen if...*, or
simply because your data is too noisy or the number of variables you are dealing with is too hight.

Statistics also plays a very important role in data visualization, as you need it to draw appropriate conclusions.
Moreover, sometimes you are forced to display some aggregate number rather than showing the raw data,
and statistics helps us choosing a good aggregation method.

## Descriptive statistics vs inferential statistics

When we perform a data analysis we should start with some well defined question,
and the subject of this question is the **population**, while the **sample**
is the set of units of our dataset.
When we collect the data, if the sample corresponds to the population we are performing a [**census**](https://en.wikipedia.org/wiki/Census), while
if the sample is a subset of the population we are performing a [**sampling**](https://en.wikipedia.org/wiki/Sampling_(statistics)).

The discipline which studies how to properly summarize the sample is called [**descriptive statistics**](https://en.wikipedia.org/wiki/Descriptive_statistics),
while [**inferential statistics**](https://en.wikipedia.org/wiki/Statistical_inference) is used to draw conclusions about the population from the sample.

In order to draw conclusions about the entire population from the sample we must make some assumption about
how our population is composed and how does our sample relates to our population,
and these assumptions are mathematically formulated into a [**statistical model**](https://en.wikipedia.org/wiki/Statistical_model).

In other words, a statistical model is a mathematically rigorous idealization of the characteristics of the population:
we are trying to catch the relevant features of some phenomenon, but it is almost always impossible
to catch *all* the features that contribute to the phenomenon:

<div class='emphbox'>
<a href="https://en.wikipedia.org/wiki/All_models_are_wrong">
All models are wrong, but some are useful.
</a>
</div>

## Statistical models

As previously stated, when we build a statistical model, our aim is to get information
about the entire population by only studying a sample.
Of course, we don't know all the characteristics of our population,
otherwise we wouldn't need to study it!
This implies that we somehow must account for our ignorance, and this can be done by 
assuming that the relevant features of our population are **randomly distributed
according to some probability distribution**, so our task is now to somehow constrain
the space of the reasonable distributions that describe our population.

At this point, if you are not familiar with these concept, you may wonder what does probability **mean**,
but this is far from being an easy question to answer,
ant this is because randomness can come into many different forms.

The measure of the spin of an electron is random, as quantum mechanics tells us that is intrinsically
impossible to know the outcome of the experiment with.

Also tossing a coin is a random event, although this could be in principle determined by Newton's laws.
In fact, even a small variation of the initial position of the coin or of its initial speed can dramatically change
its trajectory.

Tomorrow's value of Microsoft's financial stock is a random quantity too,
but this kind of ignorance is subjective, as we might know some relevant information
that we might think will affect the price.

These three examples describe three very different situations, so you should be not surprised that
there is more than one answer to the question "what probability *means*".

However, there is much more agreement on the answer to the question "what probability *is*",
and this is given by the branch of mathematics named [**probability theory**](https://en.wikipedia.org/wiki/Probability_theory).

## Probability theory

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
- $P(\Omega) = 1$ (the probability of any event is 1)
- $P\left(\bigcup_i E_i\right) = \sum_i P\left(E_i\right) \forall \{E_i\}  : E_i \cap E_j = \emptyset \forall i \neq j $ (the total probability of disjoint events is the sum of their probabilities)

We won't dig deeper in this mathematical discussion, at least for now, as an entire mathematical course would be necessary.

## Interpretations of probabilities

At this point, we can refine our previous discussion, and we can say that, in statistical inference, we generally
assume that we are seeking the probability $$P$$ which best describes our population by looking into a set of possible
probability distributions $$\mathcal{P}\,,$$ where the probability fulfills the criteria defined above.
What one generally assumes is that the space of the possible probability distributions is indexed by a vector of parameters
$$\theta$$ of dimension $$n\,,$$ so that 

$$\mathcal{P} = \{ P_\theta | \theta \in \Theta \}_\theta $$

But let us go back to our original question: what does the probability mean? Most statisticians will nowadays agree that there are
two possible interpretations of the probability, and depending on the context either one or the other might be more appropriate.

- In the frequentist interpretation, the probability represents the frequency of occurrence of an events by repeating the experiment an infinite amount of times
- In the Bayesian interpretation, the probability of an event represents the beliefs that the event will occur

The two interpretations are hiding a different treatment for the above parameter $$\theta$$

<div class='emphbox'>
In the frequentist interpretation, $\theta$ is a number (or vector of numbers), while in the Bayesian one it is a random variable itself.
</div>

This implies two very different approaches to the solution of the inferential problem.
In the frequentist approach one must find a suitable method to get information about the parameter.
These methods are very easy for simple problems, but they may soon become cumbersome as the question
becomes less trivial of if the model gets more and more involved.
In the Bayesian approach, on the other hand, one must rely on Bayes theorem (which we will discuss later in the blog)
to draw conclusions on the parameter, and this always requires a certain amount of effort, but this method
is the same regardless on the problem.

## Conclusions

We gave an overview to some fundamental branches of statistics, and we briefly discussed what probability is and how does
it relate to the real world depending on the interpretation we rely on.
In the future posts, we will take a closer look at the different methods of both frequentist statistics and Bayesian statistics.

