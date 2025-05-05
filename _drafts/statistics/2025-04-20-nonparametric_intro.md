---
layout: post
title: "Nonparametric models"
categories: /statistics/
subcategory: "Non-parametric models"
tags: /nonparametric_intro/
date: "2025-04-20"
section: 7
# image: "/docs/assets/images/perception/eye.jpg"
description: "When your focus in on flexibility"
---

There are situations where a simple model is not appropriate for your purpose.
In these cases, you may decide and use a model with a number of parameters
which is not fixed a priori, but it is rather decided depending on the data.
In this section we will discuss this kind of model, and we will focus
on the following families:

- Gaussian Processes (GP)
- Dirichlet Processes (DP), in particular Dirichlet Process Mixtures (DPM)
- Basis function expansions, in particular splines
- Bayesian Additive Regression Trees (BART)

These model are generally more flexible than the models discussed up
to now. They are also however more involved, and consequently harder
to understand. 
With respect to parametric models, non-parametric models are more likely to overfit,
so you should be careful when using them.