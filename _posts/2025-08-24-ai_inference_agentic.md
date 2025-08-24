---
layout: post
title: Proposal AI infernece + Agentic AI
date: 2025-08-24 16:40:16
description: Reasoning capability comparison of open-source and closed source models in math.
tags: inference, agentic
categories: tech
giscus_comments: true
---


## Introduction

Agentic AI is a term describing that an AI is considered as a whole system and can think and act in an environment autonomously. One prominent example is that Agentic AI is an evolution of conversational AI, enabling systems to act more autonomously and proactively. While traditional conversational AI focuses on natural language interactions, agentic AI adds the capability for systems to set goals, make decisions, and take actions to achieve those goals with limited human intervention.

One essential component of these AI systems is Large language models (LLMs). LLMs are usually in the backend, which needs to be trained (pre-trained) and then tuned to a specific task (post-training) [5]. Pre-training and post-training take a lot of effort in data and training, and computing power, and as a result, the models become larger and larger. 

In 2024, Snell et al. [1] stated that “Scaling LLM Test-Time to Compute Optimally Can Be More Effective than Scaling Model Parameters”.  Brown et al. [2] went a step further: At inference, a model should make more than one attempt at a problem. Similarly, Hao et al. [3] introduced “COCONUT” (Chain of Continuous Thought).

![image](assets/proposal_inference/image.png)

Fig.: Inference will become increasingly important in the future. At inference (aka test-time), the model could make many decisions that have not been discovered. (Image source: https://upaspro.com/inference-time-scaling-vs-training-compute) 

Recently, a first attempt at AI safety assessment has been evaluated [4] on closed-source models. They also proposed a reinforcement-based approach to mitigate safety issues. For mitigating AI safety issues can be mitigated with post-training [5]. [7] states that expanding reasoning with safety improves generalization. [8] found out that increasing inference-time computation improves the adversarial robustness of reasoning LLMs in many cases in terms of reducing the rate of successful attacks.


## Research Question
What is the AI safety risk in inference scaling in Agentic AI? E.g., mistakes.
How can those be mitigated?
How can AI be human-aligned?
Research Aims and Objectives
Investigate current methods (post-training and at inference): Evaluate them and generate a dataset, if needed. 
Develop an evaluation environment, e.g., extend AgentBench.
Based on the evaluations and benchmarks, develop your method to mitigate AI safety issues.
(Probably too futuristic: evaluate and align multi-agent systems. Agent A does not have the same alignment as agent B.)

## References
[1] Snell C, Lee J, Xu K, Kumar A. Scaling LLM test-time to compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314. 2024 Aug 6.
[2] Brown B, Juravsky J, Ehrlich R, Clark R, Le QV, Ré C, Mirhoseini A. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787. 2024 Jul 31.
[3] Hao S, Sukhbaatar S, Su D, Li X, Hu Z, Weston J, Tian Y. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769. 2024 Dec 9.
[4] Qiu R, Li G, Wei T, He J, Tong H. Saffron-1: Towards an Inference Scaling Paradigm for LLM Safety Assurance. arXiv preprint arXiv:2506.06444. 2025 Jun 6.
[5] Kumar K, Ashraf T, Thawakar O, Anwer RM, Cholakkal H, Shah M, Yang MH, Torr PH, Khan FS, Khan S. Llm post-training: A deep dive into reasoning large language models. arXiv preprint arXiv:2502.21321. 2025 Feb 28.
[7] Kumarage T, Mehrabi N, Ramakrishna A, Zhao X, Zemel R, Chang KW, Galstyan A, Gupta R, Peris C. Towards safety reasoning in LLMs: AI-agentic deliberation for policy-embedded CoT data creation. arXiv preprint arXiv:2505.21784. 2025 May 27.
[8] Zaremba W, Nitishinskaya E, Barak B, Lin S, Toyer S, Yu Y, Dias R, Wallace E, Xiao K, Heidecke J, Glaese A. Trading inference-time compute for adversarial robustness. arXiv preprint arXiv:2501.18841. 2025 Jan 31.
[9] Lu S, Bigoulaeva I, Sachdeva R, Madabushi HT, Gurevych I. Are emergent abilities in large language models just in-context learning?. arXiv preprint arXiv:2309.01809. 2023 Sep 4.
