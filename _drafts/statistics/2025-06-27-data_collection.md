---
categories: /statistics/
date: 2025-06-27
description: Choosing how to collect data
layout: post
section: 10
subcategory: Experiments
tags: /problem/
title: Data collection

---



Now that you precisely stated the question you want to answer, you must choose how
to collect your data.

## Study types

There are many kinds of studies and many study classifications.
One very broad distinction is between **experimental** studies and
**observational** studies.
In the first case you treat your units and measure the outcome,
in the latter you simply observe different populations and compare them.
Causal conclusions drawn from experiments are generally considered more valid
than the ones drawn from observational studies, but this is not always possible,
as an experimental study might imply some ethical or practical issues.


A more refined subdivision is both based on the randomization criterion
and on the fact that the researcher assigns the treatment.
We will stick
to the following naming convention:

| Treatment \ Randomization     | Randomized            | Not randomized |
|-------------------------------|-----------------------|----------------|
| **Researcher assigns treatment** | Randomized experiment | Quasi-experiment|
| **Researcher does not assign treatment** | Natural experiment    | Observational study |

This classification is not universally accepted, so whenever
you read about quasi experiments and natural experiments, make sure
you understand what the author means by them.


### A (hopefully) clarifying example

Let us try and clarify the above concept with an example, and
let's assume that we want to assess if riding bike is more dangerous than
walking.

In a randomized experiment, we would recruit a group of people,
randomly divide them into two groups and let them travel either
by walking or by riding a bike, depending on the assigned group,
measuring the number of injured for each group.

In an observational study, we would simply collect the number of injured persons
while walking and compare them with the number of injured persons while
riding a bike. Of course, there might be tons of reasons why
we observe a different number of injuries in the two groups.
As an example, poorer people have a lower chance of having a bike,
and the economic status might be related with the degree of knowledge
of the traffic laws, so the number of injuries is confounded by
the economic status.

In a quasi-experimental design, we could recruit a group of people
and give them a bike. As control group, we could use people who enrolled
late in the experiment, and ask them not to use the bike.
In this case, the treatment is assigned by us (we are giving a bike),
but the control group is not randomly assigned.

A natural experiment would require an external event to occur in order to be feasible.
As an example, if today the major of a town forbids using the bike,
we could try and perform a natural experiment by comparing the change in number of injured
persons in that town with the number of injuries in a nearby town before and after
the introduction of the law.
Of course, we must also analyze the differences between the two towns,
and exclude the presence of any systematic effect which could invalidate
our result.
If we are confident enough that the residence town is a random variable,
then we can define our study as a natural experiment.

### Further classification criteria

A study is named **census** if is performed on the entire population,
while it is named **survey** when it's performed on a subsample.
Studying the entire population is obviously better than only using a subsample,
but this often has many practical drawbacks, and surveys are often preferred
to censuses.
This naming is generally referred to questionnaire studies, but the same naming
can be also applied to any kind of study.

At the beginning, we will focus on experiments, and only in the future
we will discuss observational studies.
Experiments have a very long history, but the systematic study on the different
kinds of experiments began at the beginning of the 20th century thanks
to sir Ronald Fisher, one of the most influential statisticians
of the modern statistics, and probably the man who's considered the
father of modern statistics by many researchers.

Most of the methods we will discuss here have been originally
proposed in ["The Design of Experiments"](https://home.iitk.ac.in/~shalab/anova/DOE-RAF.pdf),
a breakthrough textbook where the principles and methods of the design of experiments
are collected and explained in great detail.

## Population, frame and sample

In any survey, once you defined the population, you must choose a sampling
frame, which will be the data source you will use to sample your units.
The sampling frame can be different from the population: there might be units
which does not belong to your population, missing units or duplicate units.
It is important to reduce this misalignment as much as possible in order
to have an unbiased estimate, and it's also important to figure out
which are the possible source of misalignment and which are your strategies
to handle them.

As an example, if you are performing a study on the students of your school,
you might have sick students which will be missing from your frame
or visiting students which should not be included in your frame.

Once you choose your frame, you should draw a sample from it, and whenever
possible this should be randomly in order to avoid any source of bias.

<svg width="350" height="300">
    <circle cx="190" cy="150" r="110"  stroke-width="4" fill="blue" />  
    <circle cx="150" cy="150" r="110" stroke-width="4" fill="steelblue"  />
    <circle cx="105" cy="150" r="40"   stroke-width="3" fill="blueviolet" />
    <circle cx="190" cy="150" r="110" stroke="gray"  stroke-width="3" fill="none" />
    <text x="80" y="150" fill="black">SAMPLE</text>
    <text x="170" y="150" fill="black">FRAME</text>
    <text x="245" y="90" fill="red">POPULATION</text>
</svg>

## Principle of DOE

As previously stated, for the moment we will stick to experimental studies.
The design of any experiment rely on three main principles:
- Randomization
- Replication
- Blocking

Randomization is a broad concept, and it is needed to eliminate
or at least reduce the possibility of a bias in the experiment.

Let us assume that you are studying two versions of the same ingredient
for a chemical process, and you first run 5 runs with version A,
you then run other 5 runs with version B.
Let us also imagine that, after each run, you clean your
instrumentation, but that a small amount of residual remains after each run,
negatively affecting the performances of the next runs.
This could impact your conclusions on the performances of version B.
It is therefore important to assign a random order to the testing order
of the 10 runs.

Randomization plays a central role when it comes to evaluate the validity of
a study.
A classical example is given by the Doll and Hill lung cancer study,
where the authors used an observational study to investigate the relationship
between tobacco usage and lung cancer.
The author concluded that there was a relationship between the phenomena,
but being not a randomized experiment, these conclusions has been criticized
for many years, since other factors could have induced this relationship.
As an example, some author suggested that the same genetic factor
was determining both the willing to smoke and the increased cancer risk.
If the same relationship between smoking and lung cancer
were observed in a randomized study, the genetic relationship would be excluded,
since the treatment assignment (smoke/don't smoke) would be random, therefore
it must be independent on the genetics.

There is sometimes confusion between random assignment, random sampling
and randomization as generally meant in the design of experiment
field.
Random sampling means to randomly choose the study units
from the population, and as we will discuss later, this helps
in generalizing the study results from the sample to the population.
Random assignment is the procedure to randomly assign the unit
to one of the treatment group, and it's a specific kind of randomization.
Finally, randomization is a broad concept which requires to randomly
choose any factor which might affect the outcome.
Examples are the treatment assignment, the treatment order or the 
assignment of any other factor which might be relevant.
Randomization is needed in order to reduce the probability
that any systematic effect which has not been accounted in the analysis
invalidates the conclusions of the study.

Since measurements might be affected by variations, it is important
to quantify the amount of variability and identifying its
sources, and replication is therefore needed.
Replication should not be confused with measurement repetition,
since in replication also the entire preparation procedure should
be repeated.

<br>

> "Block what you can, randomize what you cannot" 
> 
> George Box

<br>

Blocking is a related concept, and it consists in fixing all the quantities
which can affect the experiment result in order to reduce the variability
in the comparison.
As an example, if we are studying the effect of a therapy,
and we suspect that the sex might affect the effect, 
we might consider using sex as a blocking variable.

Unfortunately, statistics is not enough when we talk about experiment,
and the process under study should always be under control.
In the abstract example we made, the chemical residual caused a drift
in the performances, and this kind of issue should be always avoided.

## Randomization

### Simple random sampling

There are many randomization methods, and now we will discuss
some of the simplest methods.
Let us assume you want to select $K$ units out of a group of $N$ individuals.
The simplest method is the **simple random sampling with replacement**,
and it consists into drawing $K$ independent numbers going from $0$ to $N-1$[^1]

[^1]: We are still pythonists, aren't we?

```python
import numpy as np
rng = np.random.default_rng(42)

K=20
N=100

srswr = rng.choice(range(N), size=K)
srswr
```
<div class="code">
array([ 8, 77, 65, 43, 43, 85,  8, 69, 20,  9, 52, 97, 73, 76, 71, 78, 51,
       12, 83, 45])
</div>

This method is very easy to understand and explain. It has however
the drawback that the same unit might be selected more than once,
reducing the size of our sample.

Another easy-to-understand method is the toss of a coin (Bernoulli)
or the roll of a die (Categorical)
to decide to which group a unit will be assigned.
The above example could be reformulated by assigning to each individual
a probability of 0.2 to be selected

```python
bern = rng.choice(range(2), size=N, p=[0.8, 0.2])
bern
```

<div class="code">
array([0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
</div>

The main issue of this method is that the difference
between the size of the selected sample and the desired sample size (20)
might be large.

```python
np.sum(smpl)
```

<div class="code">
15
</div>

A better strategy is to use the **simple random sample without replacement**
which can be implemented by drawing a random number for each of the $N$
elements and selecting the $K$ smaller elements.

```python
srswor = (np.argsort(rng.uniform(low=0, high=1, size=N))<K).astype(int)
srswor
```
<div class="code">
array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
</div>

The above result is named mask, and it's an array of size $N$ where
there is a 1 for each selected unit.
The above method can be immediately generalized to split $N$ units
into $n$ groups, we can therefore use it to randomly split the
sample into the different treatment groups.
Assuming that $N = K*n\,,$ we can in fact order the units
according to the random numbers and assign the units from the first
to the $K$-th to the first group, the units from the $K+1$-th
to the $2K$-th to the second group and so on.

## Other sampling methods
Sometimes it is more practical to first group units and then to
only select some of the groups, either by selecting all the units
in the groups or by sampling from them (two stage sampling).
As an example, if you need to perform a survey to students,
it might be cumbersome to randomly select 100 students from all the schools
into your city, and it might be more reasonable to only select 5 schools
and only select 20 students for each of the selected schools.
This method is named **cluster sampling**, and it was very popular
when face-to-face surveys were more common.
Now, thanks to the web surveys, this method is less used, but it might still
be useful.

The last method, and probably the one which ensure that your
sample is the more representative as possible, is the 
**stratified sampling**.
Let us assume that you want to perform a survey on 20 employees
of a company.

They are divided as follows:


| Occupation | Gender | Number | Percentage |
|------------|--------|--------|------------|
| Secretary  | Male   | 25     | 10         |
| Secretary  | Female | 75     | 30         |
| Worker     | Male   | 100    | 40         |
| Worker     | Female | 50     | 20         |

In order to perform a proportional stratified random sample,
you should simply sample the 10% of your 20 employees from the first
stratum (subgroup), the 30% of your 20 employees from the second
stratum, the 40% of your 20 employees from the second
stratum and the 20% of your 20 employees from the  last
stratum, so you should randomly choose 2 males secretaries,
6 female secretaries, 8 male workers and 4 female workers.

In order to perform an analysis on some data obtained from a stratified
random sample, we can compute the relevant variable for each stratum
and weight the result according to the stratum proportion.

This is not the only stratification scheme you might use.
Another common scheme is to sample the same number of units from each
stratum, and this would result in a smaller variance in the inference
of this stratum. This method has however worst performance
on any global (*i.e.* averaged over the strata) inference.

This is the simplest kind of stratified random sampling,
and more advanced methods are possible, but we won't discuss them in this post.

## An additional warning

Data collection is never a purely statistical question, and a strong
domain knowledge is often needed to properly perform this task.
It is therefore mandatory to collaborate with a domain expert
in order to avoid pitfalls in this phase, as no
statistical model can give you good results when the underlying data
is not good enough.
This principle is often stated as

<div class="emphbox">
Garbage in, garbage out.
</div>