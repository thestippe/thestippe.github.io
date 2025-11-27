---
categories: /other/
up: /other
date: 2026-04-20
description: Getting reliable data from people
layout: post
section: 10
subcategory: Other random thoughts
tags: /complex/
title: Introduction to questionnaires
published: false
---

AI is often developed in order to assist, support or sometimes even replace
domain experts in specific tasks.
This implies that the data scientist has to compare AI-generated data
with human-generated data, and while the DS is supposed to
have a good understanding of AI-generated data, 
many of us 
are not prepared to handle human-generated data.

This is because analyzing human-generated data
generally requires a basic understanding of how humans think
and act. In other terms, if you want to properly deal with
this kind of data, you should have a good understanding of psychology.

The most common tool we use to collect data from people is by using
some kind of questionary, and questionnaires design is a 
broad multidisciplinary field, which involves psychologists,
UX experts, programmers, statisticians and other domain experts.

In the next section, I want to discuss some of the issues
you could face when dealing questionaries.
This does not want to be a course in psychology or UX design, but
it rather wants to be a warning about the fact that you could
need to consult someone with a deeper understanding of the topic,
otherwise you could end up doing a lot of useless work.

People change their mind, they sometimes feel 
uncomfortable when answering questions, they might lack
attention or forget stuff, and all these issues might undermine your data collection
process.

We will try and discuss what we should keep in mind when
designing a questionnaire, and we will try and go from the details
of the single answer to a broad overview of the question design.

## Question types

There are many possible question types, and each type
can be represented in more than one way.

### Binary values

In the simplest case, we have two mutually exclusive answers,
such as yes or no.

<form>
  <fieldset>
    <legend>Do you own a mobile phone? </legend>
    <div>
      <input type="radio" id="contactChoice1" name="contact" value="email" />
      <label for="contactChoice1">Yes</label>
      <input type="radio" id="contactChoice2" name="contact" value="phone" />
      <label for="contactChoice2">No</label>
    </div>
    <div>
      <button type="submit">Submit</button>
    </div>
  </fieldset>
</form>

### Nominal values

### Ordinal values

### Numeric values

### Open-ended question

## Question formulation

## Order of the questions
