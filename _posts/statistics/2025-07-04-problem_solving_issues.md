---
categories: /statistics/
date: 2025-07-04
description: If something can go wrong, it probably will
layout: post
section: 10
subcategory: Experiments
tags: /problem/
title: Things that could go wrong
---




Now that we understood the structure of the PPDAC workflow, let us take
a look at the possible issues that could undermine our task.

## Type III errors

Giving the right answer to the wrong question is known as **type III error**,
and this kind of error can make a great research work completely useless.

When you perform a type III error, you generally focus on the wrong
aspect of the general problem,

This generally happens because we lack the domain knowledge when we start
planning, so the probability to perform this kind of error can be reduced
by talking with domain expert and reading papers on the research topic.

Moreover, you should keep in mind the big picture whenever you decide which
actions you plan to perform to solve your problem.


## Validity issues

Validity quantifies the strength of the evidence supporting the conclusions
of the study. When two studies find opposite results, the one with higher
validity is generally considered more trustful.
There are many factors affecting the validity of a study, and here
we will discuss four of the most important and general types of validity:

- Construct validity
- Statistical validity
- Internal validity
- External validity

### Construct validity

Measures as the physical ones generally are closely related to the
abstract construct they want to quantify.
There is little doubt that the quantity that you read on a metric tape represents
the length of the object you are measuring.

Things can however be more tricky in other fields such as psychology or social sciences,
where the quantity of interest are as abstract as the well-being of someone
or his/her stress.
In these cases it might happen that some measure does not properly quantify
the construct of interest, and in this case you have a poor construct validity.

The same kind of issue might also arise in other fields such as medicine, where a certain medical
test does not properly correlate with the golden standard.
As an example, a high body temperature might not be the best way to assess the presence
of an infection.


### Statistical conclusion validity

The statistical conclusion validity refers to the appropriateness and correctness of the
statistical procedures used in the research.
The statistical conclusion validity is undermined by mathematical errors, but also
by the violations of the assumptions of the model.

If you are using a test which assumes normality, and you observe a large
skewness might affect the statistical validity of your research.
Using robust methods ensure a higher statistical validity.

Another practice which negatively affects the statistical conclusion
validity is not taking into account the proper type of the variable
of interest, *e.g.* assuming normality for ordinal variables.

Finally, the well known p-value hacking is a procedure which
seriously undermines the statistical conclusion validity of the study.
This practice occurs when you perform multiple tests on the same
data without properly taking into account the multiplicity.

### Internal validity

A study has an internal validity if the causal conclusions drawn for the
study population are appropriate.
If the study has been properly designed and conducted in order to
answer the research questions of the research, then the study has internal validity.

The presence of any statistical bias is a threat to the internal validity
of the study.


### External validity

A study has internal validity if the results of the study can be generalized
to the target population.
If the study population is representative of the target population,
then study can be generalized. A way to ensure external validity
is to make sure that the study population can be considered a random
sample of the target population.


## Conclusions

Problem-solving can be complex, many things can go wrong and even one
error can undermine the usefulness of your research.
Here we analyzed the main reasons which could lead you to a wrong solution
to your problem.


## Suggested readings

- <cite>Royall, R. (2017). Statistical Evidence: A Likelihood Paradigm. Regno Unito: CRC Press.</cite>

<!--
The first step in the PPDAC cycle is the problem identification,
and if you do not correctly perform this step, you might end up with
a beautiful but useless research work.

Giving the correct answer to the wrong question is known as **type III error**,
and this kind of error is more frequent than what you might expect.

<br>

> "Better a poor answer to the right question than a
good answer to the wrong question."
> 
> John Tukey

<br>

As an example, let us assume that you delivered some AI solution,
and the client complains about the quality of the predictions performed by
your AI.
You run your tests on your software, and the tests look fine.
You then answer that your software is fine.
In this case you performed a type III error: instead of asking yourself if
your test coverage is good enough for your purposes, you verified if
your software was able to pass your tests.

## Statistical significance vs substantive significance

A case where type III error often occurs is when the researcher
focuses on the statistical significance rather than on the substantive significance.
Statistical significance gives us information about the implausibility
of a hypothesis, it tells us nothing about the relevance of the effect on
the real world.

Let us assume that you want to increase the incomes from your website.
You therefore decide that you want to increase the number of clicks on the "buy" button
on your home page, where the client can start the buying procedure on your market,
and you sell a variety of products.

You design a new layout for your home page, and you perform
a well-designed A/B test on the two pages.
If you find that your new layout has a significantly larger number of clicks,
you can be quite confident that there is a larger amount of audience which starts the buying procedure.
What you still don't know is
1. how large is the increase of people who starts to buy (is it one per day or one per year?)
2. how many of them actually concluded the buying procedure (if it's lower, then you might even experience an economic loss)
3. what did they actually buy (are the users buying cheaper products with the new web page?)


## Suggested readings

- <cite> Mitroff, I. I., Silvers, A. (2010). Dirty Rotten Strategies: How We Trick Ourselves and Others Into Solving the Wrong Problems Precisely. US: Stanford Business Books.</cite>