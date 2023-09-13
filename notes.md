---
layout: page
title: "List of the notes"
permalink: /notes
---

{% assign postlist = site.posts | reverse %}

## Introductory material

Here you will find the ABC of Bayesian statistics with PyMC

{% for p in postlist %}
        {% assign cat = p.categories | first %}
        {% if cat contains "course/intro" %}
1. [{{p.title}}]({{p.tags}})
        {% endif %}
{% endfor %}

## Composite models

In this section we will collect somehow more advanced models.

{% for p in postlist %}
        {% assign cat = p.categories | first %}
        {% if cat contains "course/composite" %}
1. [{{p.title}}]({{p.tags}})
        {% endif %}
{% endfor %}

## Appendices

{% for p in postlist %}
        {% assign cat = p.categories | first %}
        {% if cat contains "course/appendices" %}
1. [{{p.title}}]({{p.tags}})
        {% endif %}
{% endfor %}
