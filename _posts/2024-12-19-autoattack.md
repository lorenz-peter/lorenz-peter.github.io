---
layout: post
title: AutoAttack
date: 2024-12-19 16:40:16
description: 
tags: paper
categories: adversarial-examples
giscus_comments: true
---

# AutoAttack for Adversarial Robustness

## Introduction

Adversarial training is about robustify a neural network against adversarial attacks.

- More details: [here](https://adversarial-ml-tutorial.org/adversarial_training).
- Link to [AutoAttack](https://arxiv.org/pdf/2003.01690). 

## Key insights

Authors do not argue that AutoAttack [1] is the ultimate adversarial attack but rather that it should become the minimal test for any new defense, since it reliably reaches good performance in all tested models, without any hyperparameter tuning and at a relatively low computational cost.

3 weaknesses of PGD:

1. Fixed step size: suboptimal, even for convex problems this does not guarantee convergence, and the performance of the algorithm is highly influenced by the choice of the value. [2]
2. Agnostic of the budget: The loss plateaus after a few iterations, except for extremely small step sizes, which however do not translate into better results. Judging the strength of an attack by the number of iterations is misleading. [3]
3. Unaware of the trend: Does not consider whether the optimization is evolving successfully and is not able to react of this. 
Authors present an automatic scheme fixing this issue. 


# References

- [1] [Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks, ICML, 2020.](https://arxiv.org/abs/2003.01690)
- [2] [Logit Pairing Methods Can Fool Gradient-Based Attacks, NeurIPSw, 2018.](https://arxiv.org/abs/1810.12042)
- [3] [On Evaluating Adversarial Robustness, arxiv, 2019.](https://arxiv.org/abs/1902.06705)