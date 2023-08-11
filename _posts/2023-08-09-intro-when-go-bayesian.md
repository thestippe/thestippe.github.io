---
layout: page
title: "Why (and when) should you go for Bayesian"
---
I feel quite a pragmatic person, so I think that one should choose the tool depending on the needs rather than by relying on some personal believes.
Bayesian statistics allows to build custom and structured models by simply specifying the data generating process.
The model can be divided into two parts:
- The likelihood $P(y \vert \theta)$, which determines how the data we want to model $y$ are generated given the parameter(s) $\theta$.
- The priors $P(\theta)$, which specifies our hypothesis about the distribution of the parameters of the model.

The only mathematical requirements for both the likelihood and for the priors is that they are non-negative and sum up to one.
There is a huge literature about the model building, and one can easily start by using one of the already available models and adapt it
to the problem under study.

Once that the model is built one samples the entire posterior probability distribution,
which is determined by means of Bayes theorem.

$$ P(\theta \vert y) = \frac{P(y \vert \theta) P(\theta)}{P(y)} \propto  P(y \vert \theta) P(\theta) $$

Here $\propto$ means proportional to, which means equal up to some irrelevant
multiplicative positive constant, which is the inverse of the probability of the data
$P(y)\,.$

This makes Bayesian statistics very attractive if you are building a statistical
model to make a decision, as you can easily make inference about any kind of quantity
regarding your model, while this is not true if you only have a point estimate 
or an interval estimate, as it happens in Machine Learning.

Moreover, Bayesian statistics is easily interpretable: what you are doing
is simply to use the data to update your initial believes.

So why is not everyone using it? There are many reasons for this, some of them
are historical, other are more pragmatic.

On the one hand, the possibility to easily implement a numerical simulation
and to run it within a reasonable amount of time is relatively recent and not
yet spread outside the statistical community. However, things are changing,
as the misuse and misinterpretation of tools of frequentist statistics is considered 
one of the main reason for the so-called _reproducibility crisis in science_
and a **proper** use of Bayesian methods is considered a valid alternative to those
tools [^1] [^2] [^3] [^4].

The major practical drawback of Bayesian statistics is that you need to run your
simulation thousands of times, and this may take some time if the number
of parameters in your model is large.
Thus, I do not reccomend you to use Bayesian statistics if your task is a simple
and fast fit-predict problem.

There is another drawback, and this is where these notes come into play:
building a model without a basic knowledge about model building and assessment
is not an easy task. There are many beautiful online courses about Bayesian
statistics in R, but either you already know R, or you need to learn Bayesian
statistics _and_ R.
This blog is written to make this task simpler for anyone who has a basic knowledge
about Python, and since Python is the most widely spread programming language
in the World, I hope I will help a lot of people.

I hope you enjoyed,

Stefano


[^1]: This is somehow a misleading name, as this crisis is not only affecting academia, but it is a problem in industry too.
[^2]: See [this article on Nature](https://www.nature.com/articles/533452a)
[^3]: See [this other article of Nature](https://www.nature.com/articles/520612a)
[^4]: Of course, Bayesian statistics can be misused too, but there are few very clear
guidelines from the academic community which will make this less likely to happen.
Moreover, in most cases, a problem in your model will show up in a problem
in your simulation, and this makes Bayesian inference less error-prone
than frequentist inference.
In fact, when talking about frequentist statistics or machine learning,
most of the time what you are computing is either an optimization problem or
the average of some quantity,
and an eventual problem in this kind of task may be quite hard to spot.
