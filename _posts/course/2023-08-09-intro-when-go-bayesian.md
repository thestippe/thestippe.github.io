---
layout: page
title: "Why (and when) should you go for Bayesian"
---
I feel I am quite a pragmatic person, so I prefer choosing my tools depending on my needs rather than by relying on some personal believes.
Bayesian statistics allows you to build custom and structured models by simply specifying the data generating process.
The model can be divided into two parts:
- The likelihood $p(y \vert \theta)$, which determines how the data you want to model $y$ are generated given the parameter(s) $\theta$.
- The priors $p(\theta)$, which specifies your initial hypothesis about the distribution of the parameters of the model.

The only mathematical requirements for both the likelihood and for the priors is that they are non-negative and sum up to one.
There is a huge literature about the model building, and you can easily start by using one of the already available models and adapt it
to your problem.

Once that the model is specified you can use $PyMC$ or any other Probabilistic Programming Language
sample the entire posterior probability distribution,
which is determined by means of Bayes theorem.

$$ p(\theta \vert y) = \frac{p(y \vert \theta) p(\theta)}{p(y)} \propto  p(y \vert \theta) p(\theta) $$

Here $\propto$ means proportional to, which means equal up to some multiplicative positive constant,
where by constant we mean independent on $\theta$.
The constant $p(y)$ can be fixed by requiring that $p(\theta \vert y)$ is normalized to one:

$$1 = \int d\theta p(\theta | y) = \frac{1}{p(y)}\int d\theta p(y \vert \theta) p(\theta)$$

so

$$ p(y) = \int d\theta p(y|\theta)p(\theta)\,. $$

The fact that you sample the entire probability distribution $p(\theta \vert y)$
makes Bayesian statistics very attractive if you are building a statistical
model to make a decision, as you can easily make inference about any kind of quantity
regarding your model.
This is rarely possible if you only have a point estimate or an interval estimate, as it happens in Machine Learning.

Moreover, Bayesian statistics is easily interpretable: what you are doing
is simply to use the data to update your initial believes.
In fact, in the Bayesian interpretation, $p(\theta)$ represents your opinion about the possible
values that $\theta$ may take before you make an experiment and observe $y\,.$
On the other hand $p(\theta \vert y)$ represents your updated opinion about the value of $\theta$
after the experiment.

So why is not everyone using it? In my experience there are multiple reasons, some of them
are historical, others are more pragmatic.

First of all, the possibility to easily implement a numerical simulation
and to run it within a reasonable amount of time is relatively recent and not
yet spread outside the statistical community.
Bayesian statistics was the only available framework up to the end of the nineteenth 
century, and in has been largely abandoned at the beginning of the last century,
when Fisher and his collaborators developed frequentist statistics.
People in fact considered Bayesian statistics very difficult,
as the normalization factor $p(y)$
can only be computed for a very limited number of models
(the so-called _conjugate_ models).
De Finetti, Savage, Jeffreys and others tried to convince
people to abandon the frequentist framework as they did not considered the
frequentist interpretation satisfactory, but they never managed to convince
the majority of the community.
Things changed when, during the Manhattan project, Metropolis, Von Neumann, Ulam and
others invented the
Markov Chain Monte Carlo methods, and this allowed physicists and later statisticians
too to 
draw random samples from an arbitrary probability distribution.
Moreover, nowadays, the misuse and misinterpretation of tools of frequentist statistics is considered 
one of the main reason for the so-called _reproducibility crisis in science_
and a **proper** use of Bayesian methods is considered a valid alternative to those
tools [^1] [^2] [^3].
Of course, Bayesian statistics can be misused too, but there are few very clear guidelines from the academic community which will make this less likely to happen.
Moreover, in most cases, a problem in your model will show up in a problem in your simulation, and this makes Bayesian inference less error-prone than frequentist inference.
In fact, when talking about frequentist statistics or machine learning, most of the time what you are computing
is either an optimization problem or the average of some quantity.
Since what you obtain from this kind of procedure is a number rather than a sample,
in this kind of task may be quite hard to spot.

However, the major practical drawback of Bayesian statistics is that you need to run your
simulation thousands of times, and this may take some time if the number
of parameters in your model is large.
Thus, I do not reccomend you to use Bayesian statistics if your task is a simple
and fast fit-predict problem.

There is another drawback, and this is where these notes come into play:
building a model without a basic knowledge about model building and assessment
is not an easy task. There are many beautiful online courses about Bayesian
statistics in R, which is the most common programming language between statisticians.
The choice of the programming language implies that either you already know R, or you need to learn Bayesian
statistics _and_ R.
This blog is written to make this task simpler for anyone who has a basic knowledge
about Python, and since Python is the most widely spread programming language
in the World, I hope I will help a lot of people.

I hope you enjoyed,

Stefano


[^1]: This is somehow a misleading name, as this crisis is not only affecting academia, but it is a problem in industry too.
[^2]: See [this article on Nature](https://www.nature.com/articles/533452a)
[^3]: See [this other article of Nature](https://www.nature.com/articles/520612a)