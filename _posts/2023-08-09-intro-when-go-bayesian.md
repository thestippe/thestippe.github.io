---
layout: default
title: "Why (and when) should you go for Bayesian"
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    TeX: {
      equationNumbers: {
        autoNumber: "AMS"
      }
    },
    tex2jax: {
    inlineMath: [ ['$', '$'] ],
    displayMath: [ ['\[', '\]'] ],
    processEscapes: true,
  }
});
MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

I feel quite a pragmatic person, so I think that one should choose the tool depending on the needs rather than by relying on some personal believes.
Bayesian statistics allows to build custom and structured models by simply specifying the data generating process.
The model can be divided into two parts:

The likelihood 
$P(y \vert \theta)$, which determines how the data we want to model are generated.

The priors $P(\theta)$, which specifies our hypothesis about the value of the parameters of the model.

$$ a = \xi y^2 $$
