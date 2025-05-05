---
layout: post
title: "Fitting complex models"
categories: /statistics/
subcategory: "Other random thoughts"
tags: /complex/
date: "2025-05-08"
section: 10
# image: "/docs/assets/images/perception/eye.jpg"
description: "When the going gets tough"
---

The aim of this post is a little bit different from all the previous
posts, and it's to illustrate how we can use PyMC to fit complex models.
To do so, we will use the Diebold-Li model, which is a model commonly
used in finance to fit the yield of bonds and other fixed income securities.
A bond is characterized by its duration, which usually goes from one
month to 30 years, and they are sold on a regular basis.
The US federal bank provides the daily values of all the US treasure bonds,
and our aim is to fit these curves.
The dataset can be obtained from
[this link](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2025).
