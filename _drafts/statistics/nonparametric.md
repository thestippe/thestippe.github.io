---
layout: post
title: "Introduction to non-parametric models"
categories: /statistics/
subcategory: "Non-parametric models"
tags: /nonparametric_intro/
date: "2024-03-02"
section: 7
# image: "/docs/assets/images/perception/eye.jpg"
description: "Letting the number of parameters vary"
---

Non-parametric models are becoming more and more popular in Bayesian
statistics, as they are able to accurately reproduce complex data
and patterns.
In this section we will introduce some of the most popular
non-parametric models, namely splines and Dirichlet processes-related
models.

When dealing with these models, one should keep in mind that
the Bernstein-Von Mises theorem does not hold, so one cannot
approximate frequentist confidence intervals with Bayesian
credible intervals.