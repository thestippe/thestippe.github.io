---
layout: page
title: "List of the notes"
permalink: /notes
---

{% assign postlist = site.posts | reverse %}

<br>
## Introductory material

Here you will find the ABC of Bayesian statistics with PyMC

{% for p in postlist %}
        {% assign cat = p.categories | first %}
        {% if cat contains "course/intro" %}
1. [{{p.title}}]({{p.tags}})
        {% endif %}
{% endfor %}

<br>
## Composite models

In this section we will collect somehow more advanced models.

{% for p in postlist %}
        {% assign cat = p.categories | first %}
        {% if cat contains "course/composite" %}
1. [{{p.title}}]({{p.tags}})
        {% endif %}
{% endfor %}

<br>
## Appendices

Some more mathematical stuff.

{% for p in postlist %}
        {% assign cat = p.categories | first %}
        {% if cat contains "course/appendices" %}
1. [{{p.title}}]({{p.tags}})
        {% endif %}
{% endfor %}

<br>
## Various

Ideas and models related to other topics.

{% for p in postlist %}
        {% assign cat = p.categories | first %}
        {% if cat contains "course/various" %}
1. [{{p.title}}]({{p.tags}})
        {% endif %}
{% endfor %}

## Dataviz

Let's talk about data visualization

{% for p in postlist %}
        {% assign cat = p.categories | first %}
        {% if cat contains "dataviz" %}
1. [{{p.title}}]({{p.tags}})
        {% endif %}
{% endfor %}
