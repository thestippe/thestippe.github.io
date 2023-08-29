---
layout: page
title: "List of the notes"
permalink: /notes
---

{% assign postlist = site.posts | reverse %}

## Introductory material

{% for p in postlist %}
        {% assign cat = p.categories | first %}
        {% if cat contains "course/intro" %}
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
