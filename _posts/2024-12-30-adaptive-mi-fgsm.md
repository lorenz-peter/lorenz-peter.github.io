---
layout: post
title: AdaMSI-FGM
date: 2024-12-30 16:40:16
description: 
tags: paper
categories: adversarial-examples
giscus_comments: true
---

# On the Convergence of an Adaptive Momentum Method for Adversarial Attacks

## Introduction

This paper [1] aims to fill the gap between empirical evaluations and theoretical fundamentals of MI-FGSM. 
MI-FGSM improves the itertive FGSM (i-FGSM or BIM) by adding a momenumt which helps to overcome local minima and hence the adversarial examples transfer better. 
However, it is a sign-based attack method, where the sign gives an bound of the magnitude of the gradient step. 
Sign-based methods fail to converge to the optimum in convex settings. 
To address these concerns, the authors propose a novel method (AdaMSI-FGM), which incorporates both an innovative adaptive momentum parameter with monotonicity assumptions and an adaptive step-size scheme that replaces the sign operation.

## Key insights

- Sign-based attack methods are well-known and this better showed there is still research to be done.
- The sign method can be replaced with an adaptive update step.
- Derive a regret upper bound for general convex functions.

# References

- [1] [On the Convergence of an Adaptive Momentum Method for Adversarial Attack, 2024, AAAI.](https://ojs.aaai.org/index.php/AAAI/article/view/29323)