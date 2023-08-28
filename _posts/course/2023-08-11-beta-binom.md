---
layout: page
title: "Beer and the Beta-Binomial model"
categories: course/intro/
tags: /beta_binom/
---
I love beer, and whenever I have a free day I brew. As you probably know, beer is made
with water, malt, hop and yeast. One of the most important things to do in order
to produce a good beer is to have a good quality yeast, and one of the metrics
used to quantify the goodness of the yeast is the **yeast viability**, which corresponds to the percentage of alive cells in your yeast.
This procedure is time consuming, as you must count by hand the number of dead
and alive cells in a sample, so it is usually performed with small samples. It is therefore important to quantify the uncertainties in your estimate.


Unfortunately, most home-brew textbooks will only give you a way to
estimate the mean yeast viability, and you may get fooled by your count and think that
you are working with a good yeast while you simply overestimated the yeast viability.
If you want to know more about how to experimentally count the yeast cells,
you can take a look to [this](https://escarpmentlabs.com/blogs/resources/crop-pray-count-yeast-counting-guide)
link, where the procedure to count the yeast cells is illustrated.

In the standard procedure, one has a $5\times 5$ grid and one counts the alive
cells and the death ones, where one can distinguish the cells thanks to the
Trypan Blu which will color the death cells.
A simulated example of what one will see is shown below:

![Alt text](/docs/assets/images/beta_binom/yeast_count.jpg)

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

$$ \theta_{hb} = \frac{n_a}{n_a + n_d}  $$

```python
theta_hb = alive / total
```
> 0.886


This is a quick solution, however we cannot associate any uncertainty to this number
for the moment.


## The frequentist statistician's way

A frequentist statistician would first of all setup a model for this problem.
The state of each cell can take two values:

$$
y_i = 
\begin{cases}
1 \text{ (alive) } & \text{ with probability } \theta \\
0 \text{ (death) } & \text{ with probability } 1-\theta
\end{cases}
$$

If we assume that the probability of being alive of each cell is independent on the probability of the remaining cells
of being alive and that the probability is the same for each cell, we have that the probability of finding $y$ alive cells out of $n$ total counted cells must follow a binomial distribution:

$$p(y|p, n) \propto \theta^{y} (1-\theta)^{n-y}$$

which can be written as

$$ y \sim Binomial(\theta, n) $$

where the binomial distribution has probability mass

$$ p(y | p, n) = \binom{n}{y} \theta^y (1-\theta)^{n-y} $$

and 

$$y = \sum_{i=1}^n y_i$$

and $ \binom{n}{y} = \frac{n!}{y!(n-y)!}$ is a multiplicative normalization factor.
Once the model is built, we want to find $p$ such that the $p(y | \theta, n)$ is maximum, namely the *Maximum Likelihood Estimator* or MLE for the
sake of brevity.
$p(y | p, n)$ is a positive quantity for $\theta \in (0, 1)$, and this allows us to take its logarithm, which is a monotone increasing function, and 
this implies that the maximum of $\log p$ is the maximum of $p\,.$

$$ \log p(y | \theta, n) \propto y \log \theta + (n-y) \log(1-\theta) $$

$$ \frac{\partial \log p(y | \theta, n)}{\partial \theta} = \frac{y}{\theta} + \frac{n-y}{\theta-1} $$

$$ \left. \frac{\partial \log p(y | \theta, n)}{\partial \theta}\right|_{\theta=\hat{\theta}} = 0 \Rightarrow \frac{y}{\hat{\theta}} = \frac{n-y}{1-\hat{\theta}} \Rightarrow \hat{\theta}(n-y) = (1-\hat{\theta}) y
\Rightarrow \hat{\theta} n = y$$

Which gives us, again, $\hat{\theta} = \frac{y}{n}\,,$ which is the same value that we got by using the home-brewer textbook's way.

We can easily verify that it is a maximum:

$$ \frac{\partial^2 \log p(y | \theta, n)}{\partial \theta^2} = -(n - y)/(\theta - 1)^2 - y/\theta^2 $$

$$ \left. \frac{\partial^2 \log p(y | \theta, n)}{\partial \theta^2}\right|_{\theta=\hat{\theta}} =
-\frac{n^3}{y (n - y) }$$

and the last quantity is always negative, for $0<y<n\,.$

The frequentist statistician, however, knows that his estimate for the alive cell
fraction is not exact, and he would like to provide an uncertainty interval
associated to the estimate.
He can use the central limit theorem, which says that, if $n$ is large, then the binomial distribution can be approximated with the normal distribution
with the same mean and variance of the binomial distribution, which corresponds to $\mu = n\hat{\theta}$ and $\sigma^2= n\hat{\theta}(1-\hat{\theta})\,.$
He would use this theorem to provide the $95\%$ Confidence Interval for this distribution.

For a normal distribution with mean $\mu$ and variance $\sigma$ the $95\%$ CI
is given by

$$ \mu \pm z_{1-0.05/2}\sigma $$

where $z_{1-0.05/2}=1.96$ is the $0.975$ normal quantile.
So we can easily obtain the $95\%$ confidence interval for $\theta$ as

$$ \frac{\mu \pm \sigma}{n} = \hat{\theta} \pm  z_{1-0.05/2} \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{n}} 
 = \hat{\theta} \pm  1.96 \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{n}}  = [0.81, 0.96]$$


The calculation is quite straightforward, but one should pay a lot of attention in giving the correct interpretation to this interval.
In the frequentist paradigm, one imagines to repeat the experiment many times, and what one can say is that, by doing this,
if the confidence interval is constructed with the procedure given above, in the $95\%$ of the repetitions it will contain the true
fraction of alive cells.
However, it doesn't tell us anything about the confidence we have that the fraction of alive cells is in the interval $[0.81, 0.96]\,.$

**For the frequentist statistician, the probability that the true value lies inside [0.81, 0.96] is either 1 if it is inside 
or 0 if it is not inside, but he cannot say which one is correct!**

This fact is often misinterpreted, even by many researchers and data scientists.

## The Bayesian rookie's way

The Bayesian statistician would take the same likelihood for the model, however in his framework the parameter $\theta$ is
simply another random variable, and it is described by some other probability distribution $p(\theta)$ namely by the **prior** associated
to the parameter $\theta\,.$

$\theta$ can take any value between 0 and 1, but he has no preference about any value, so he assumes that $\theta$ is distributed
according to the uniform distribution over $[0, 1]\,.$

$$ \theta \sim Uniform(0, 1) $$

```python
with pm.Model() as beta_binom_model:
    theta = pm.Uniform('theta')
    y = pm.Binomial('y', p=theta, n=total, observed=alive)
    trace = pm.sample(random_seed=rng)

az.plot_trace(trace)
```


![Alt text](/docs/assets/images/beta_binom/trace_bb.jpg)

He used PyMC to sample $p$ many times according to its posterior probability distribution,
obtained by using the Bayes theorem

$$ p(\theta | y, n) \propto p(y | \theta, n) p(\theta)$$

and the sampled values are those shown in the figure.
The details about how does PyMC's sampler works will be explained in a future post,
as well as the main methods to exclude problems in the sampling procedure.


```python
az.plot_posterior(trace)
```
![Alt text](/docs/assets/images/beta_binom/posterior_bb.jpg)

We can see that the mean is very close to the MLE value, and the (Bayesian)
$95\%$ CI (which corresponds to the two printed numbers) is close to
 the frequentist one too.
However in this case the interpretation is straightforward:

**the Bayesian statistic simply updated his/her initial guess for $p$ by means of Bayes' theorem.**

For the Bayesian statistician there is the $95\%$ of chance that the true
value of $p$ lies inside the $95\%$ CI associated to $\theta$.

Another major advantage of the Bayesian approach is that we did not had to rely
on the Central Limit Theorem, which only holds if the sample is large enough.
The Bayesian approach is always valid, regardless on the size of the sample.

## The wise Bayesian's way

The wise Bayesian would follow an analogous procedure, he would however
take the less informative prior as possible, where a strongly informative
prior is a prior which influences a lot the posterior probability distribution.
The uniform distribution is not a very informative distribution.
However, as we will show, we can even choose a less informative prior, namely
the **Jeffreys' prior** for the binomial distribution

$$ \theta \sim Beta(1/2, 1/2) $$

where the Beta has pdf

$$ p(\theta | \alpha, \beta) = \frac{1}{B(\alpha, \beta) } \theta^\alpha (1-\theta)^\beta$$

and $B(x, y)$ is the Beta function.
However, he knows he knows he must pay a lot of attention, as often -but not
in this case- the Jeffreys' prior is not a proper prior
(it cannot be integrated to one) [^1].


```python
with pm.Model() as beta_binom_model_wise:
    theta = pm.Beta('theta', 1/2, 1/2)
    y = pm.Binomial('y', p=theta, n=total, observed=alive)
    trace_wise = pm.sample(random_seed=rng)

az.plot_trace(trace_wise)
```

![Alt text](/docs/assets/images/beta_binom/trace_bb_wise.jpg)

```python
az.plot_posterior(trace_wise)
```

![Alt text](/docs/assets/images/beta_binom/posterior_bb_wise.jpg)

As we can see, this result is almost identical to the previous one.
In the Bayesian framework one can, and should, investigate the goodness
of his results by trying out different priors and assess how
much does the results on his/her inference depend on the choice of the priors.

## Conclusions and take-home message

- PyMC allows you to easily implement Bayesian models.
- In many cases Bayesian statistics offers results which are more transparent than their frequentist counterparts. We have seen this for a very simple model, but this becomes even more evident as the complexity of the model grows.
- You can apply Bayesian statistics to any kind of problem, even home-brewing!

In the [next](/count_data/) example we will apply Bayesian statistics to study
data which can take more than two values.

[^1]: This topic will be discussed in a future post. For the moment, if you are curious, you can take a look at the [Wikipedia](https://en.wikipedia.org/wiki/Jeffreys_prior#) page.
