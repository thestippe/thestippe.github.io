---
layout: post
title: "Spatial survival analysis"
categories: /other/
up: /other
tags: /spatial_survival/
# image: "/docs/assets/images/gis/point_patterns/posterior_xi.webp"
description: "Space as a risk factor"
date: "2025-01-24"
published: false
---

This post will be somehow different from all the previous ones, since we will not
illustrate a specific kind of model, but we will rather try and see how
to gradually add structure to a model in order to improve it.
The model choice is in fact driven by what we need to do with the model,
and in this case we will try and determine if there is correlation
between the spatial location and the leukemia incidence in the UK.
We will use the [Leukemia Survival Data](https://rdrr.io/cran/spBayesSurv/man/LeukSurv.html),
which is a part of the R spBayesSurv package, and it has originally analyzed
in [Modeling Spatial Variation in Leukemia Survival Data](https://www.jstor.org/stable/3085818)
by Henderson, Shimakura and Gorst.


```python
import xarray as xr
import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import pymc_bart as pmb
from sksurv.nonparametric import kaplan_meier_estimator
from SurvSet.data import SurvLoader

rng = np.random.default_rng(20251031)

kwargs = dict(nuts_sampler='nutpie', draws=3000, tune=3000, chains=4, random_seed=rng)

kwargs_base = dict(draws=3000, tune=3000, chains=4, random_seed=rng)

loader = SurvLoader()

df, ref = loader.load_dataset(ds_name='LeukSurv').values()

df.head()
```


```python
df['fac_district'] = pd.Categorical(df['fac_district'])
df['bin_sex'] = (df['fac_sex']=='F').astype(int)

df['censoring'] = [None if event else time for time, event in zip(df['time'], df['event'])]

```