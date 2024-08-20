---
layout: post
title: "Experiment analysis with many blocking variables"
categories: /statistics/
subcategory: "Advanced models"
tags: /experiment_analysis/
date: "2024-06-03"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Dealing with more complicated experiments"
section: 5
---

In the last post, we discussed how to design experiments
either without blocking variables, or with a single blocking variable.
In this post we will extend the previous discussion to a larger number of
blocking variables.

## Full factorial design

Let us now consider the extension of the RBD to $k>2$ blocking variables,
and let us assume that each of them has $n$ possible levels.
In a full factorial design, you run the experiment for all the possible combinations
of the different levels of the blocking variables.
Since the number of possible combinations is $n^k\,,$ the number of levels and
of variables is usually indicated as $n^k\,.$
Using this kind of design has of course pros and cons: the design is easier to
perform and to communicate, and you can measure all the possible interactions
among the variables.
It requires however many different runs, and this can be both time and resources demanding.

In this section, we will re-analyze the experiment proposed as an exercise at page
68 of "Design and analysis of experiments with R" taken from [an article by Stuart Hunter](
https://www.tandfonline.com/doi/abs/10.1080/08982118908962680),
and here's the dataset

|    |   ethanol |   air_fuel_ratio |   co2_emission_1 |   co2_emission_2 |
|---:|----------:|-----------------:|-----------------:|-----------------:|
|  0 |       0.1 |               14 |               66 |               62 |
|  1 |       0.1 |               15 |               72 |               67 |
|  2 |       0.1 |               16 |               68 |               66 |
|  3 |       0.2 |               14 |               78 |               81 |
|  4 |       0.2 |               15 |               80 |               81 |
|  5 |       0.2 |               16 |               66 |               69 |
|  6 |       0.3 |               14 |               90 |               94 |
|  7 |       0.3 |               15 |               75 |               78 |
|  8 |       0.3 |               16 |               60 |               58 |

The author replicated each measure twice, and he determined the $CO_2$
emissions of a fuel depending on the amount of ethanol used
and on the air/fuel ratio.

As in the original article, we will use a quadratic model with a linear interaction term.

```python
df_in = pd.read_csv('./data/co2.csv')
df1 = df_in[['ethanol', 'air_fuel_ratio', 'co2_emission_1']]
df2 = df_in[['ethanol', 'air_fuel_ratio', 'co2_emission_2']]

df = pd.concat([df1.rename(columns={'co2_emission_1': 'co2_emission'}),
df2.rename(columns={'co2_emission_2': 'co2_emission'})], axis=0)
```

Let us first normalize the regression variables,
and let us assume a linear interacting model for the outcome

```python
x0 = (df['ethanol']-0.2)/0.2
x1 = (df['air_fuel_ratio']-14)/2
x = np.array([np.ones(len(x0)), x0, x1, x0*x1, x0**2, x1**2])

with pm.Model(coords=
              {'col_id': 
                   ['intercept', 'ethanol', 'af_ratio', 'interaction', 'ethanol_sq', 'af_ratio_sq'],
                      'obs_id': df.index}) as ffmodel:
    X = pm.Data('X', x, coords=['col_id', 'obs_id'])
    beta = pm.Normal('beta', mu=0, sigma=100, dims=['col_id'])
    mu = pm.math.dot(beta, X)
    sigma = pm.HalfNormal('sigma', 200)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=df['co2_emission'])
    idata_rbd = pm.sample(nuts_sampler='numpyro', draws=5000, tune=5000, chains=4, random_seed=rng)

az.plot_trace(idata_rbd)
fig = plt.gcf()
fig.tight_layout()
```

![](/docs/assets/images/statistics/experiment_design/ff_trace.webp)

The trace seems fine, but let us take a closer look at the values of the $\beta$ coefficients.

```python
az.plot_forest(idata_rbd, var_names=['beta'])
```

![](/docs/assets/images/statistics/experiment_design/ff_forest.webp)

Both the variables play an important role, and the presence of an interaction
between them is quite strong.
Let us verify if we are able to reproduce the data.

```python
df_out = df[['ethanol', 'air_fuel_ratio']].drop_duplicates()
x0new = (df_out['ethanol']-0.2)/0.2
x1new = (df_out['air_fuel_ratio']-14)/2
xnew = np.array([np.ones(len(x0new)), x0new, x1new, x0new*x1new, x0new**2, x1new**2])
ffmodel.add_coords({'sample_new': np.arange(len(x0new))})

with ffmodel:
    Xnew = pm.Data('Xnew', xnew, coords=['col_id', 'sample_new'])
    mu_new = pm.math.dot(beta, Xnew)
    y_new = pm.Normal('y_new', mu=mu_new, sigma=sigma)
    ppc = pm.sample_posterior_predictive(idata_rbd, var_names=['y_new'])
    
df_out['mean'] = ppc.posterior_predictive['y_new'].mean(dim=('draw', 'chain')).values
df_out['q03'] = ppc.posterior_predictive['y_new'].quantile(q=0.03, dim=('draw', 'chain')).values
df_out['q97'] = ppc.posterior_predictive['y_new'].quantile(q=0.97, dim=('draw', 'chain')).values

fig = plt.figure()
ax = fig.add_subplot(111)
sns.scatterplot(df, x='ethanol', hue='air_fuel_ratio', y='co2_emission')
legend = ax.legend()
sns.lineplot(df_out, x='ethanol', y='mean', hue='air_fuel_ratio', ax=ax, legend=None)
sns.lineplot(df_out, x='ethanol', y='q03', hue='air_fuel_ratio', ax=ax, ls=':', legend=None)
sns.lineplot(df_out, x='ethanol', y='q97', hue='air_fuel_ratio', ax=ax, ls=':', legend=None)
fig.tight_layout()

```

![](/docs/assets/images/statistics/experiment_design/ff_ppc.webp)

The model seems to reproduce the observed data quite well.
As an exercise, verify the performances of the linear model with the above model.



## Latin square design

In the randomized block design, one can only control for one factor, but it may also be the case
that you need to control for more than one factor.
The latin square design is useful when you need to control for two factors,
but you don't have enough resources to perform a full factorial design.
This design can be visualized by drawing an $n\times n$ table, where each row corresponds
to the level of one factor, the other level is represented by the column, and each matrix element
is represented by a number $1,...,n$ or by a (latin) letter,
and correspond to the treatment group.
In a latin square, no letter can appear twice in any row or column.
By using the latin square, you assign each possible treatment to each row and column.

All the possible $2\times 2$ latin squares are

$$
\begin{pmatrix}
1 & 2 \\
2 & 1 \\
\end{pmatrix},
\begin{pmatrix}
2 & 1 \\
1 & 2 \\
\end{pmatrix}
$$

while a possible $3\times 3$ latin square is

$$
\begin{pmatrix}
1 & 2 & 3 \\
2 & 3 & 1 \\
3 & 1 & 2 \\
\end{pmatrix}
$$

In order to obtain a random $n\times n$ latin square, you can simply use the following
function

```python
import random

def latin_square(n):
    row = list(range(n))
    mat = [row]
    for i in range(1, n):
        mat += [row[i:]+row[:i]]
    random.shuffle(mat)
    return mat
```

We will analyze the dataset provided at [this link](https://rdrr.io/github/kwstat/agridat/man/bridges.cucumber.html)
where we want to assess the performances of a set of cucumber cultivars, and the
experimental setup is a latin square design experiment repeated into two different locations.

We will use a hierarchical model for the latin square components, plus a linear term 
to account for the location.
We will only take a non-zero mean for the cultivar parameter, while we will
assume a zero mean for the row and column components.

```python
df_latin_square = pd.read_csv('https://raw.githubusercontent.com/kwstat/agridat/main/data/bridges.cucumber.txt', sep='\t')
df_latin_square['gen'] = pd.Categorical(df_latin_square['gen'])
df_latin_square['loc'] = pd.Categorical(df_latin_square['loc'])
```

```python
with pm.Model() as lsmodel:
    xi = pm.Normal('xi', mu=0, sigma=50)
    mu = pm.Normal('mu', mu=0, sigma=100)
    rho = pm.Exponential('rho', 0.05, shape=(3))
    sig_alpha = pm.Normal('sig_alpha', mu=0, sigma=1, shape=(4))
    sig_beta = pm.Normal('sig_beta', mu=0, sigma=1, shape=(4))
    sig_gamma = pm.Normal('sig_gamma', mu=0, sigma=1, shape=(4))
    alpha = pm.Deterministic('alpha',mu + rho[0]*sig_alpha)
    beta = pm.Deterministic('beta', rho[1]*sig_beta)
    gamma = pm.Deterministic('gamma', rho[2]*sig_gamma)
    sigma = pm.Exponential('sigma', 0.01)
    tau = xi*df_latin_square['loc'].cat.codes+alpha[df_latin_square['gen'].cat.codes] + beta[df_latin_square['row']-1]+ gamma[df_latin_square['col']-1]
    y = pm.Normal('y', mu=tau, sigma=sigma, observed=df_latin_square['yield'])

with lsmodel:
    idata_ls = pm.sample(nuts_sampler='numpyro', draws=5000, tune=5000, chains=4, random_seed=rng,
                        target_accept=0.9)

az.plot_trace(idata_ls)
fig = plt.gcf()
fig.tight_layout()
```

![](/docs/assets/images/statistics/experiment_design/ls_trace.webp)

In the above model, $\alpha$ corresponds to the average cultivar yield
for the first location, where the average is taken over the rows and columns.
We can also identify $\mu$ with the average yield for the first location.

Another reasonable choice would have been to parametrize the model in order to obtain
the difference between the first cultivar and the other cultivars as parameters,
and we leave the implementation of this model as an exercise.

Let us now take a closer look at the values of the different $\alpha$s.

```python
az.plot_forest(idata_ls, var_names=['^alpha'], filter_vars='regex')
```

![](/docs/assets/images/statistics/experiment_design/ls_forest.webp)

We can easily assess the performances of the cultivars with respect to the benchmark cultivar:

```python
fig, ax = plt.subplots(nrows=3, figsize=(6, 7))
for i in range(1, 4):
    sns.kdeplot(idata_ls.posterior['alpha'][:, :, i].values.reshape(-1)-idata_ls.posterior['alpha'][:, :, 0].values.reshape(-1),
               ax=ax[i-1])
    ax[i-1].set_title(r"$\alpha_"+str(i)+r"-\alpha_0$")
fig.tight_layout()
```

![](/docs/assets/images/statistics/experiment_design/ls_alpha_diff.webp)

## Conclusions

We have discussed how to adapt some model used in experimental
design with more than one blocking variable to make them Bayesian, and again we did so by using PyMC.


## Suggested readings

- <cite>Box, G. E. P., Hunter, J. S., Hunter, W.G. (2005). Statistics for experimenters: design, innovation, and discovery. Wiley.</cite>
- <cite>Lawson, J. (2014). Design and Analysis of Experiments with R. CRC Press.<cite>


```python
%load_ext watermark
```

```python
%watermark -n -u -v -iv -w -p xarray,numpyro,jax,jaxlib
```

<div class="code">
Last updated: Mon Aug 19 2024
<br>

<br>
Python implementation: CPython
<br>
Python version       : 3.12.4
<br>
IPython version      : 8.24.0
<br>

<br>
xarray : 2024.5.0
<br>
numpyro: 0.15.0
<br>
jax    : 0.4.28
<br>
jaxlib : 0.4.28
<br>

<br>
seaborn   : 0.13.2
<br>
pandas    : 2.2.2
<br>
arviz     : 0.18.0
<br>
matplotlib: 3.9.0
<br>
numpy     : 1.26.4
<br>
pymc      : 5.15.0
<br>

<br>
Watermark: 2.4.3
<br>
</div>