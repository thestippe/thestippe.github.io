---
layout: post
title: "A/B testing"
categories: /statistics/
tags: /testing/
# image: "/docs/assets/images/perception/eye.jpg"
description: "An example of statistical test"
---

In this post we will see how to perform A/B testing in python as
well as the correct interpretation of the results of the test.

## An issue with mailing lists

I help as volunteer in a no-profit organization, and one of my tasks is to prepare the weekly newsletter.
Some of us was not happy with the preview layout, so we came out with some minor modifications.
Apparently the modifications were fine, but before changing the layout we decided to perform a test.
Fortunately our mail service provides a tool for A/B testing, so we decided to split our audience in two
equal blocks.
The tool randomly assign each recipient into one of two (or more) groups with a given probability.
Then a version of the mail is sent to one group, while another version of the mail is sent to the other group,
and by looking at how many subjects opened the newsletter (or performed some other action)
you can decide which version is better.

## Randomization

The procedure of randomly assigning an individual to a group is named randomization.
Randomizing your groups is really important, otherwise you can't exclude that the difference in the behavior is
due to the selection criteria you choose.
As an example, assume that we assigned all the new subscribers to group A, while old subscribers are assigned to 
group B.
Suppose that your historical audience is rather adult, but that you recently went to many schools and many students
subscribed to your newsletter in the last month.
You then do your test, and observe that the group A opened more newsletter, so you concluded that it's better version A.
You can't however exclude that group A opened more newsletter because young people look more frequently
to emails, so the result may be strongly biased by your selection criteria.

Of course, randomization may give you unbalanced groups, and even by randomizing the samples you may end up
with a group with an average age which is much more than the average age of the other group.
In order to exclude this situation you should compare the distributions of the ages as well as any other potentially
relevant quantity, named confounder, and verify that the two groups are similar.
While this practice is usually performed in clinical studies, in our case we don't have this data due to privacy issues,
so we are forced to skip this test.

<!--
Spiegare cosa si intende per test, randomizzazione, causalità e manipolazione
-->

## Before running the experiment

In order not to bias the experiment, it is fundamental to decide how to run the experiment *before*
the experiment itself.
In this post we will discuss the frequentist approach, while the Bayesian one will be discussed in a future one.

In both cases we assume that the probability of an individual belonging to group $i$ to open the letter is given by $0\leq \theta_i \leq 1\,.$
This means that, given the number of delivered emails of the $i$-th group $n_i$, the number of opened mails follows

$$
y_i \sim Binom(n_i, \theta_i)
$$

What we want to determine is if the opening probability after the update $\theta_2$
is greater than the opening probability before $\theta_1\,.$

## The frequentist point of view

Our null hypothesis $H_0$ is that the average opening rate of the old layout will be greater than the one of the new
layout, and we would like to exclude this case.
When you work with tests you have four different possibilities

- $H_0$ is true and you don't reject it: true negative
- $H_0$ is true and you reject it: false positive - type I error
- $H_0$ is false and you don't reject it: false negative - type II error
- $H_0$ is false and you reject it: true positive

The probability of having a type I error is named **statistical significance** and it is usually indicated with the letter $\alpha$.
On the other hand, the probability of having a type II error is indicated with $\beta\,,$ while $1-\beta$ is
named the **statistical power**.

When you build the test, you usually fix $\alpha$ to some small number, then you choose your sample size so that
your power is large enough for your assumed effect.
A value of $\alpha=0.05$ is generally used, and sometimes it is justified, sometimes it is not.
In our case, a probability to reject a true null hypothesis of 1 against 20 is good enough,
so we will stick to this case.
Now it comes the question of choosing beta. We have a fixed number of subscribers, and they are roughly 11000,
but we may perform the test more on more than one newsletter.
An a priori estimate of the increase of the average opening rate is not very easy, but we may say that an increase of
a $2\%$ would be a good reason to switch to the new layout.
We know that the opening rate is roughly $0.35\,,$ so now we have all the ingredients for an estimate of the power of our test.


Notice that $y_i$ is the sum of $n_i$ independent variables, and $n_i \gg 1\,,$ so we can safely use the result
of the central limit theorem, which states that the sum of a large number of iid random variables
with mean $\mu$ and variance $\sigma^2$ can be approximated by a normally distributed random variable
with mean $\mu$ and variance $\frac{\sigma^2}{n}\,.$
In our case, the probability that a single subscriber of the group $i$ that received the email opens it
is a Bernoulli random variable with mean $\theta_i$ and variance $\theta_i (1-\theta_i)\,.$

This implies that we can approximate
$$ \frac{y_i}{n_i} $$ with a normal variable with mean $\theta_i$ and variance $\frac{\theta_i (1-\theta_i)}{n_i}\,.$
The two sample Student's t-test allows you to test if two normally distributed samples have equal mean,
so in the large $n$ approximation we can use this test to verify our hypothesis.

## How does a test work

In this section we will sketch the main ideas behind statistical testing from a frequentist perspective.
Let us now assume that we have a large sample $X_1,\dots,X_n$ of $n$ random variables *with known variance $\sigma^2$*, and we want to determine if
the sample average is compatible with $\mu$ within a significance $\alpha\,.$
In order to do so, we construct the variable

$$Z = \frac{\bar{X}-\mu}{\sigma/\sqrt{n}}$$

and we already know that, if $X_i \sim \mathcal{N}(\mu, \sigma)\,,$ then

$$
Z \sim \mathcal{N}(0, 1)
$$

What we have to do is to verify how likely is that our observed $Z$ is compatible with $\mathcal{N}(0, 1)\,.$
We reject the null hypothesis $H_0$ if the probability of the observed value for $Z$ is too far away from the center of the normal
distribution, namely if

$$
\Phi(Z) > 1-\alpha/2
$$

or

$$
\Phi(Z) < \alpha/2
$$

where $\Phi$ indicates the normal cumulative distribution function with zero mean and unit variance.
The above conditions can be summarized with

$$ \Phi(\left|Z\right|)>\alpha/2$$

or

$$
2(1-\Phi(\left|Z\right|))<\alpha
$$

If, instead of comparing a sample with a number, we want to compare two samples with known
variance, we can follow exact the above procedure, since the difference between two normal variables
$\mathcal{N}(\mu_1, \sigma_1)$ and $\mathcal{N}(\mu_2, \sigma_2)$
is distributed as $\mathcal{N}(\mu_1-\mu_2, \sqrt{\sigma_1^2 + \sigma_2^2})$

Our main issue is that, in our case, the variance is not knows, so we must define

$$Z = \frac{\bar{X}}{S/\sqrt{n}}$$

where our estimated variance is given by the unbiased estimator

$$
S^2 = \frac{1}{n-1}\sum_{i} (X_i - \bar{X})^2
$$

The new term in the denominator is no more a constant,
but it's now a random variable, and it's distributed according to the $\chi^2$ (chi squared) distribution with $n-1$ degrees of freedom.
This implies that our new random variable $Z$ is no more normal, but it is distributed according to the so-called Student's t distribution
with $n-1$ degrees of freedom, so we must now replace $\Phi$ with the Student's t cdf in the above formulas.

Notice that we have shown how to perform a two-sided test.
In a one-sided test you should either choose the condition $1-\Phi(Z)<\alpha$ or $\Phi(Z)<\alpha\,.$
The form of the above condition has been chosen so that we always have

$$ t(Z) < \alpha$$

where $t(\cdot)$ is named the **test statistics** and its value $t(Z)$ is named the **p value**.

We should now clarify what does the above p value mean. 
If $H_0$ is true, then you expect to observe a p value smaller or equal to the observed one
a fraction of times equal to the reported p value.

## Working it out

We can now perform the simulation of the experiment.

```python
# Let's import the libraries that we will use
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st

rng = np.random.default_rng(seed=42)

pwr = []
ntot = 11000
f0 = 0.50

effect = 0.02
p1 = 0.35
p2 = p1 + effect
n2 = int(ntot*f0)
n1 = ntot-n1

alpha = 0.05

# We simulate the experiment 1000 times

for k in range(1000):
    k2 = rng.binomial(n=n2, p=p1)  # Our control sample
    k1 = rng.binomial(n=n1, p=p2)  # Our test sample
    # We convert the data in a useful format for scipy
    dt2 = [1]*k2+[0]*(n2-k2)
    dt1 = [1]*k1+[0]*(n1-k1)
    # We now test if the mean of dt2 is greater than the mean of dt1
    # We use ttest_ind since the samples are independent
    out = st.ttest_ind(dt2, dt1, alternative='greater', equal_var=False)
    # We don't assume that the two groups have equal variance
    pwr.append(out.pvalue)

# Let us now compute the power by calculating the fraction of experiments with a p value smaller than our significance
np.mean([int(elem<alpha) for elem in pwr])
```
<div class='code'>
0.701
</div>

So our chances to  reject $H_0$ when $H_0$ is false is 0.7 if the effect of the size of 0.02.
It is a satisfactory number for our purposes,
and since we don't want to waste too much time in preparing the test in the newsletter,
we won't repeat the experiment a second time.

## The results

We finally sent the test mail, and got these results:

|Group | Delivered | Opened |
|---|---|----|
| Test | 5299 | 1891 |
| Control    | 5258 | 1722 |

We can now calculate the test

```python
n2 = 5299
k2 = 1891

n1 = 5258
k1 = 1722

s2 = [1]*k2+[0]*(n2-k2)
s1 = [1]*k1+[0]*(n1-k1)

st.ttest_ind(s2, s1, alternative='greater', equal_var=False)
```

<div class='code'>
TtestResult(statistic=3.1803802487550357, pvalue=0.0007375367353631094, df=10553.261020307757)
</div>

So we can reject the null hypothesis $H_0\,,$ since our p value is smaller than $\alpha\,.$

## Conclusions

We discussed what does hypothesis testing mean and how to perform a test in python.
We have also seen the underlying assumption and the correct interpretation of the results.
In the future we will compare this method with the Bayesian approach.
Moreover, we will see what are the risks that you may take if you violate the above procedure.


<!--

### The Bayesian point of view

In the Bayesian framework, what we have to do is
- assign a prior distribution to the parameters $\theta_i$
- decide which condition on the $\theta_i$ is the null hypothesis $H_0$
- compute $\theta_i$ (we will do so by using PyMC) and verify that we didn't had numerical issues
- verify if our assumption for the priors is generous enough to accommodate the data
- verify if our condition holds

chi2 distribution and t-Student
Optional stopping
-->
