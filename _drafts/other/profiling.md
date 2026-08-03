---
layout: post
title: "Writing fast and reliable code"
categories: /other/
up: /other
tags: /hpc/
# image: "/docs/assets/images/gis/point_patterns/posterior_xi.webp"
description: "Python for High Performance scientific computing"
date: "2026-04-20"
---

When I was in academia, I spent quite a lot of time figuring out how to write good code,
and the first question is: what does **good** code means?
Reproducibility is one of the pillars of science, so having reproducible code is crucial.
This is also true when you write code for production, at least in the EU, since this is one
of the requirements of the AI act.
However, you should also be able to understand what is your code doing and what to do in order
to update it, so good code should also be reliable and readable.
We will therefore start by discussing what does it mean to have a reproducible, readable and reliable code
Scientific computing, however, can be really intensive from a computational point of view,
so we will also discuss how to improve performances.

A very nice introductory talk about this topic is [this talk by Sebastian Witowsky at EuroPython](https://www.youtube.com/watch?v=YjHsOrOOSuI),
and if you want to go deeper within this topic I would recommend you to take
a look at [the webpage of the Advanced Scientific Programming in Python summer school](https://aspp.school/archives),
where you will find a very nice set of lectures and notebooks, 
and at the [High Performance Python book](https://www.google.it/books/edition/High_Performance_Python/kKjgDwAAQBAJ?hl=it&gbpv=0),
which is a little bit outdated but still a great starting point.


## Virtual environments
Isolating the dependencies is the first step to write reproducible code,
and unless you need to deploy your code, a virtual environment is the easiest way to do so.
There are many tools to isolate the dependencies of your code inside an environment,
and here I will list the most popular ones.

### venv

As a novice, I started with [venv](https://docs.python.org/3/library/venv.html),
which comes with the Python library.
You can easily create a virtual environment with

```bash
python -m venv ./my_new_venv
source ./my_new_venv/bin/activate
```

Once you activated the environment, you can install anything inside it with

```bash
pip install stuff
```

If you need to export the environment's packages, you can run

```bash
pip freeze > requirements.txt
```

and in the requirements.txt file you will find all of your dependencies.
In order to exit from the environment, you can simply run

```bash
deactivate
```
### conda

[conda](https://docs.python.org/3/library/venv.html) is very popular is the data-science
community, and I really like it when I need to run some model in a notebook.
One great advantage of conda over venv, is that conda allows you to choose the python
version

```bash
conda create --name my_conda_venv python=3.13
conda activate my_conda_venv
```

To export the dependencies environment

```bash
conda env export > environment.yml
```

while to deactivate it

```conda deactivate```

### uv

Conda is great if you only need to install libraries, but if you are developing a package
it might not be the best option in my opinion.
In this case, you might need to split your dependencies. As an example,
it's a good idea to keep the testing libraries separated from the package dependencies,
since who installs your package does not need to test it.
Moreover, if you are building a package, you might want to separate
the dependencies of your stable release from the ones of your dev release.

[uv](https://docs.astral.sh/uv/guides/install-python/) does all the above things, and it also allows you to easily release your package to pip.

To create a new project with python 3.12 and numpy installed, you can run

```bash
uv init myproject python=3.12
uv add stuff
```

This will create a pyproject.toml file, where the package information are stored.
To run a script:

```bash
uv run script.py
```

You can then build the package with

```bash
uv build
```

and you can publish it with

```bash
uv publish
```

## Writing code

Writing readable, well documented and bug-free code is another fundamental ingredient, and
keeping in mind all the conventions can give you a good headache.
Generally speaking, it's always a good idea
to stick to [PEP 8 style](https://peps.python.org/pep-0008/).

There are many python packages which can help you adhere to the above style guidelines.
I both used [pylint](https://pypi.org/project/pylint/) and [black](https://github.com/psf/black)
in the past, and now I am trying [ruff](https://docs.astral.sh/ruff/).
There are some difference between them, but all of them are good to write decent code.
ruff can be used to automatically fix the code, whenever possible, but it requires some tuning
to include all the rules you need.
pylint is much slower, but it comes with a large set of built-in rules.
As an example, if you write a script like

```python
# script.py
import os, sys
import numpy as np

path = os.getcwd()

y =np.arange(5)


def fun(variable):
    if variable>0:
        return variable**2
    else:
        return 0
```

and (after including all the pylint rules in ruff) run

```bash
ruff check --fix
```

the script will change in

```python
import os
import numpy as np

path = os.getcwd()

y =np.arange(5)


def fun(variable):
    if variable>0:
        return variable**2
    return 0

```

and you will get an output such as

<div class="code">
D100 Missing docstring in public module<br>
--> script.py:1:1<br>
<br>
D103 Missing docstring in public function<br>
  --> script.py:9:5<br>
   |<br>
 9 | def fun(variable):<br>
   |     ^^^<br>
10 |     if variable>0:<br>
11 |         return variable**2<br>
   |<br>
<br>
Found 5 errors (3 fixed, 2 remaining).
</div>

ruff cannot add docstring on its own, but it fixed three other errors:
- it removed the unused sys import
- it removed an unnecessary else statement
- it added a blank line at the end of the code

Using any of the above tools, or any other linter, will highly increase the quality of your code.
There are however aspects which cannot be handled by these tools, and the first one is proper naming.

You should always give a meaningful name to all the variables, function, classes and to any other
object, at least as far as the objects are public.
Whenever a variable has a meaning, explain it in the variable name, and avoid
naming it `x`, `y` or with similar name, unless it's clear that you are performing regression
and that `y` is the dependent variable and `x` is the independent one.
If a function performs an operation, such as the square root, name it `square_root`
instead of `f`.
If the objects are not public, then declare them as private by adding a double underscore before
the name, *e.g.* name it `__var`.

Also writing docstrings and type hints are good habits, and you should generally stick to one
of the many available docstring styles.
So, instead of writing

```python
f = lambda x: 0 if x==0 else (1 if x==1 else f(x-1)+f(x-2))
```

it's better

```python
def fibonacci(n: int) -> int:
    """
    The n-th element of the Fibonacci succession,
    defined by
    .. math::
        f(n)=f(n-1)+f(n-2),f(0)=0,f(1)=1
    Parameters
    ----------
    n: int
        The index of the element

    Returns
    -------
    int
        The corresponding Fibonacci number
    """
    if n==0:
        return 0
    if n==1:
        return 1
    return f(n-1)+f(n-2)
```

Using an IDE can also greatly help you in writing good code, once you understand how to use it.
IDE comes with syntax checkers, code structure highlighting, debugger and much more, so coding
will be easier. There are many valuable IDEs around, and my favourite is
[PyCharm](https://www.jetbrains.com/pycharm/),
since it has been developed having Python as reference language.

## Testing code

Once you have a properly working version, you want a fast way to verify if any of the changes you did
introduced some undesired behavior, and testing has exactly this purpose.
As usual, there are many ways to write tests.

### Unit test

The first method is the python built-in testing library, [unittest](https://docs.python.org/3/library/unittest.html).
In order to implement a test, you simply have to create a subclass of `unittest.TestCase`
and implement each test as a separate method.

```python
# test_file.py
import unittest

class TestStuff(unittest.TestCase):
    
    def test_method(self):
        x = 1 + 1
        self.assertEqual(x, 2, 'Test not passed')



if __name__ == '__main__':
    unittest.main()
```

### Doctest

The second method is [doctest](https://docs.python.org/3/library/doctest.html),
which allows you to put tests within docstrings.
In order to implement a doctest, you can simply run your function in the Python console
and copy-paste the session with the proper indentation.
You can either put the doctests within the function, class or method docstring
or within the module docstring.

```python

def fib(n):
    """Returns the n-th Fibonacci number, n non-negative integer
    >>> fib(5)
    5
    >>>
    """
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        return fib(n-1)+fib(n-2)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
```

### Pytest
[pytest](https://docs.pytest.org/en/stable/)
is an extremely flexible package that makes testing easier with respect to unittest.
You can customize its behavior, but by default it will look
for any function with a name starting with the 'test' keyword within any
file starting with the 'test' keyword  and run it.

```python
# test_file.py

def my_sum(x, y):
    return x+y

def test_answer():
    assert my_sum(1, 2) == 3
```

You can run 

```bash
pytest .
```

## Code versioning

Code versioning can save you a lot of time, since it will allow you to track any change in your
code, easily revert them, having multiple versions of it and collaboratively modifying it.
And there is only one tool for code versioning, [git](https://git-scm.com/), any
other tool  is not far from being comparable to git, as long as we are talking about software.

Git requires a remote repository, and the most popular ones are
[github](https://github.com/) and [gitlab](gitlab.com/).
Both of them are great, but I prefer GitLab because GitHub is owned by Microsoft.


## Repo structuring

Structuring your repo in a logical way will make easier to search inside it, and also in this
case it's a good idea to stick to the conventions, when possible.
I found [this guide](https://docs.python-guide.org/writing/structure/)
really useful, since it's a good starting point for structuring your project.
Things are changed, as an example it's no more necessary to include setup.py
inside the project, and I would recommend using a pyproject.toml rather a requirements.txt.
However, the general structure is still valid, so I would recommend something like:

<div class="code">
README.md<br>
LICENSE.txt<br>
pyproject.toml<br>
project/<br>
tests/<br>
</div>

## Conclusions

We discussed the importance to write good, readable, well documented and maintainable code.
within your project and pytest will run all the tests within your current directory.
The above suggestions are not meant to be exhaustive, but if you have any doubt, just
remember the golden rule


> Always code as if the guy who ends up maintaining your code will be a violent psychopath
> who knows where you live.

<!---
To add next

We will start with the [three rules of optimization](https://wiki.c2.com/?RulesOfOptimization):
1. Don't!
2. Don't... yet!
3. Profile before optimizing!
--->