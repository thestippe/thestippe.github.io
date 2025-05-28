---
layout: post
title: "Frailty models"
categories: /statistics/
subcategory: "Advanced models"
tags: /survival_continuous/
date: "2025-03-24"
# image: "/docs/assets/images/perception/eye.jpg"
description: "Accounting for unknown risk factors"
section: 3
---

Frailty models have been developed to include unknown risk factors
which relates to some grouping variable in the study population.
These models are nothing but random effect models, where the random
effect is generally included as a multiplicative factor.

As every model family in Bayesian inference,
frailty models can be implemented in many flavours, and here
we will implement them in an AFT model.

## The dataset

In our analysis, we will use the [Leukemia Survival Data](https://rdrr.io/cran/spBayesSurv/man/LeukSurv.html)
where the authors analyzed te survival analysis of 1043 leukemia patients.
The authors both collected a set of prognostic factors such as age or sex
and also the residential location of each patient.

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt
import pytensor as pt
from sksurv.nonparametric import kaplan_meier_estimator
from SurvSet.data import SurvLoader

loader = SurvLoader()

rng = np.random.default_rng(65432)

kwargs = dict(tune=2000, chains=4, draws=2000, random_seed=rng)

df = loader.load_dataset('LeukSurv')['df']

df.head()
```

|    |   pid |   event |   time |   num_xcoord |   num_ycoord |   num_age |   num_wbc |   num_tpi | fac_sex   |   fac_district |
|---:|------:|--------:|-------:|-------------:|-------------:|----------:|----------:|----------:|:----------|---------------:|
|  0 |     0 |       1 |      1 |     0.205072 |     0.497244 |        61 |      13.3 |     -1.96 | F         |              9 |
|  1 |     1 |       1 |      1 |     0.285557 |     0.848953 |        76 |     450   |     -3.39 | F         |              7 |
|  2 |     2 |       1 |      1 |     0.176406 |     0.736494 |        74 |     154   |     -4.95 | F         |              7 |
|  3 |     3 |       1 |      1 |     0.244763 |     0.210584 |        79 |     500   |     -1.4  | M         |             24 |
|  4 |     4 |       1 |      1 |     0.327453 |     0.907387 |        83 |     160   |     -2.59 | M         |              7 |

```python
df['fac_district']=pd.Categorical(df['fac_districtun'].astype(int), ordered=True)
```

The district is not an ordinal variable, but the above option ensures
that the numerical code increases as the district number increases.

```python
df['log_time'] = np.log(df['time'])
df['censoring_aft'] = [None if x==1 else y for x, y in zip(df['event'], df['log_time'])]
```