---
layout: page
title: "Beer and the Beta-Binomial model"
---
I love beer, and whenever I have a free day I brew. As you probably know, beer is made
with water, malt, hop and yeast. One of the most important things to do in order
to produce a good beer is to have a good quality yeast, and one of the metrics
used to quantify the goodness of the yeast is the **yeast viability**, which corresponds to the percentage of alive cells in your yeast.
However, measuring the viability is a time consuming process, as you must
count the number of dead and alive cells in your yeast by hand.
Because of this, often the estimate is done with small samples, and due to this
it is important to quantify the uncertainties in your estimate.
Unfortunately, most home-brew textbooks will only give you very poor models to
estimate the yeast viability, and you may get fooled by your count and think that
you are working with a good yeast while you simply overestimated the yeast viability.
If you want to know more about how to experimentally count the yeast cells,
you can take a look to [this](https://escarpmentlabs.com/blogs/resources/crop-pray-count-yeast-counting-guide)
link, where the procedure to count the yeast cells is illustrated.

In the standard procedure, one has a $5\times 5$ grid and one counts the alive
cells and the death ones, where one can distinguish the cells thanks to the
Trypan Blu which will color the death cells.

![Alt text](/docs/assets/images/yeast_count.jpg)

Since counting all the cells would require a lot of time, one usually counts
five well separated squares, usually the four corner squares and the center one.
In the figure shown above:

| square | alive | death |
|------|-------|-------|
|   top left    | 16   | 2 |
|   top right     | 17   | 3 |
|   bottom left | 18   | 3  |
|   bottom right  | 11   | 0  |
|   center       |  8   | 1  |
|  **total**     | 70   | 9  |

Let us see how can we estimate the viability.
In the following, we will indicate with $n_a$ the number of alive cells (which is 70)
and with $n_d$ the number of death cells
In order to do this, let us first open our Jupyter notebook, import some libraries
and define the data

```python
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pymc as pm
import arviz as az

# Let us improve the graphics a little bit
plt.style.use("seaborn-v0_8-darkgrid")

cmap = sns.color_palette("rocket")

# For reproducibility
rng = np.random.default_rng(42)

alive = 70
death = 9
total = alive + death
```


## The home-brewer's textbook way
The home-brewer's solution is fast and simple: if we have 70
alive cells out of 79 cells, then the probability of having
and alive cell is simply

$$ p = \frac{n_a}{n_a + n_d}  $$

```python
p_naive = alive / total
```
> 0.886


This is a quick-and-dirty solution, it has however as a drawback that we have
no idea about what is the associated uncertainty to this number.


## The frequentist statistician

A frequentist statistician would first of all setup a model for this problem.
The state of each cell can take two values:

$$
y = 
\begin{cases}
1 \text{ (alive) } & \text{ with probability } p \\
0 \text{ (death) } & \text{ with probability } q=1-p
\end{cases}
$$

If we assume that the probability of being alive of each cell is independent on the probability of the remaining cells
of being alive and that the probability is the same for each cell, we have that the probability of finding $y$ alive cells out of $n$ total counted cells must follow a binomial distribution:

$$P(y|p, n) \propto p^{y} (1-p)^{n-y}$$

which can be written as

$$ y \sim Binomial(p, n) $$

where the binomial distribution has probability mass

$$ P(y | p, n) = \binom{n}{y} p^y (1-p)^{n-y} $$

and $ \binom{n}{y} = \frac{n!}{y!(n-y)!}$ is a multiplicative normalization factor.
Once the model is built, we want to find $p$ such that the $P(y | p, n)$ is maximum, namely the *Maximum Likelihood Estimator* or MLE for the
sake of brevity.
$P(y | p, n)$ is a positive quantity for $p \in (0, 1)$, and this allows us to take its logarithm, which is a monotone increasing function, and 
this implies that the maximum of $\log P$ is the maximum of $P\,.$

$$ \log P(y | p, n) \propto y \log p + (n-y) \log(1-p) $$

$$ \frac{\partial \log P(y | p, n)}{\partial p} = \frac{y}{p} + \frac{n-y}{\hat{p}-1} $$

$$ \left. \frac{\partial \log P(y | p, n)}{\partial p}\right|_{\hat{p}} = 0 \Rightarrow \frac{y}{\hat{p}} = \frac{n-y}{1-\hat{p}} \Rightarrow \hat{p}(n-y) = (1-\hat{p}) y
\Rightarrow \hat{p} n = y$$

Which gives us, again, $\hat{p} = \frac{y}{n}$

The frequentist statistician, however, knows that his estimate for the alive cell fraction is not exact, and he would like to know how much can this interval
be large.
He can use the central limit theorem, which says that, if $n$ is large, then the binomial distribution can be approximated with the normal distribution
with the same mean and variance of the binomial distribution, which corresponds to $\mu = n\hat{p}$ and $\sigma^2= n\hat{p}(1-\hat{p})\,.$
He would use this theorem to provide the $95\%$ Confidence Interval for this distribution, which is given by
$$ \hat{p} \pm  z_{1-0.05/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} 
 = \hat{p} \pm  z_{1-0.05/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}  = [0.81, 0.96]$$

where $z_{1-0.05/2}=1.96$ is the $0.975$ normal quantile.

The calculation is quite straightforward, but one should pay a lot of attention in giving the correct interpretation to this interval.
In the frequentist paradigm, one imagines to repeat the experiment many times, and what one can say is that, by doing this,
if the confidence interval is constructed with the procedure given above, in the $95\%$ of the repetitions it will contain the true
fraction of alive cells.
However, it doesn't tell us anything about the confidence we have that the fraction of alive cells is in the interval $[0.81, 0.96]\,.$

**For the frequentist statistician, the probability that the true value lies inside [0.81, 0.96] is either 1 if it is inside 
or 0 if it is not inside, but he cannot say which one is correct!**

This fact is often misinterpreted, even by many researchers and data scientists.

## The Bayesian rookie

The Bayesian statistician would take the same likelihood for the model, however in his framework the parameter $p$ is
simply another random variable, and it is described by some other probability distribution $P(p)$ namely by the **prior** associated
to the parameter $p\,.$

$p$ can take any value between 0 and 1, but he has no preference about any value, so he assumes that $p$ is distributed
according to the uniform distribution over $[0, 1]\,.$

$$ p \sim Uniform(0, 1) $$

```python
with pm.Model() as beta_binom_model:
    p = pm.Uniform('p')
    y = pm.Binomial('y', p=p, n=total, observed=alive)
    trace = pm.sample(random_seed=rng)

az.plot_trace(trace)
```


![Alt text](/docs/assets/images/trace_bb.jpg)

He used PyMC to sample $p$ many times according to its posterior probability distribution,
obtained by using the Bayes theorem

$$ P(p | y, n) \propto P(y | p, n) P(p)$$

and the sampled values are those shown in the figure.
The details about how does PyMC's sampler works will be explained in a future post,
as well as the main methods to exclude problems in the sampling procedure.


```python
az.plot_posterior(trace)
```
![Alt text](/docs/assets/images/posterior_bb.jpg)

We can see that the mean is very close to the MLE value, and the (Bayesian)
$95\%$ CI is close to the frequentist one.
However in this case the interpretation is straightforward:

**the Bayesian statistic simply updated his/her initial guess for $p$ by means of Bayes' theorem.**

Another major advantage of the Bayesian approach is that we did not had to rely
on the Central Limit Theorem, which only holds if the sample is large enough.
The Bayesian approach is always valid, regardless on the size of the sample.

## The wise Bayesian

The wise Bayesian would follow an analogous procedure, he would however
take the less informative prior as possible, where a strongly informative
prior is a prior which influences a lot the posterior probability distribution.
The uniform distribution is not a very informative distribution.
However, as we will show, we can even choose a less informative prior, namely
the **Jeffreys' prior** for the binomial distribution

$$ p \sim Beta(1/2, 1/2) $$

where the Beta has pdf

$$ P(p | \alpha, \beta) = \frac{1}{B(\alpha, \beta) } p^\alpha (1-p)^\beta$$

and $B(x, y)$ is the Beta function.
However, he knows he knows he must pay a lot of attention, as often -but not
in this case- the Jeffreys' prior is not a proper prior
(it cannot be integrated to one) [^1].


```python
with pm.Model() as beta_binom_model_wise:
    p = pm.Beta('p', 1/2, 1/2)
    y = pm.Binomial('y', p=p, n=total, observed=alive)
    trace_wise = pm.sample(random_seed=rng)

az.plot_trace(trace_wise)
```

![Alt text](/docs/assets/images/trace_bb_wise.jpg)

```python
az.plot_posterior(trace_wise)
```

![Alt text](/docs/assets/images/posterior_bb_wise.jpg)

As we can see, this result is almost identical to the previous one.
In the Bayesian framework one can, and should, investigate the goodness
of his results by trying out different priors and assess how
much does the results on his/her inference depend on the choice of the priors.

## Conclusions and take-home message

- PyMC allows you to easily implement Bayesian models
- In many cases Bayesian statistics offers results which are more transparent than their frequentist counterparts. We have seen this for a very simple model, but this becomes even more evident as the complexity of the model grows.
- You can apply Bayesian statistics to any kind of problem, even home-brewing!

[^1]: This topic will be discussed in a future post. For the moment, if you are curious, you can take a look at the [Wikipedia](https://en.wikipedia.org/wiki/Jeffreys_prior#) page.
