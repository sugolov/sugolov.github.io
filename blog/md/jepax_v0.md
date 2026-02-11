---
title: "jepax v0: an implementation of IJEPA-B training in JAX/Equinox"
date: 2026-02-11
author: "[Owen L](https://lockwo.github.io/)., Anton S."
---
*This post is a work in progress!*

## Links

**Repository:** [github.com/sugolov/jepax](https://github.com/sugolov/jepax)


## Overview
A little while ago, Owen and I got interested in JEPAs and the self-supervised approach to learning good latent representations. One theme in ongoing JEPA work are new loss regularizers: the  training setups are similar but with small augmentations to the loss that improve stability or representation properties. We set out to make `jepax` a [JAX](https://github.com/google/jax)/[Equinox](https://github.com/patrick-kidger/equinox) implementation of the self-supervised method, with the goal of a simple and modifiable codebase that enables fast iteration. 

![](images/ijepa_b.png)
<p align="center"><em><b>Figure:</b> Training loss and linear probe accuracy for IJEPA-B trained for 300 epochs on 8xA100.</em></p>

For this first release, `jepax v0`, we focused on (1) 1-to-1 configs, losses, and logging with the original PyTorch implementation and (2) a  reproduction of IJEPA-B with data parallelization on 8xA100. I think we collected a lot of interesting insights about JEPA training, which I want to describe in this blog. Some of the themes to discuss:

1. Background on JEPA and IJEPA
2. Interesting failure modes
	1. Smooth $L_1$ loss and unnormalized $L_2$
	2. Target normalization
3. JAX specific considerations

Stay tuned!