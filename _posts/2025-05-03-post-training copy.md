---
layout: post
title: Post-training
date: 2025-05-03 16:40:16
description: foundation models
tags: foundation models, post-training
categories: tech
giscus_comments: true
---


Post-training strategies like **RLHF (Reinforcement Learning from Human Feedback)** and **DPO (Direct Preference Optimization)** can also be applied to **image-generation models** (e.g., Stable Diffusion, DALL·E, Midjourney) to improve alignment, safety, and aesthetic quality. However, since images are non-textual, the methods differ slightly from those used in language models. Below is how these techniques work for image-generation AI:

---

## **1. Key Challenges for Image Models vs. Text Models**
| **Aspect**          | **Text Models (LLMs)** | **Image Models (Diffusion/VAEs)** |
|---------------------|----------------------|-------------------------------|
| **Output Type**     | Discrete tokens      | Continuous pixel space        |
| **Preference Feedback** | Easier (rank text responses) | Harder (subjective, multi-dimensional) |
| **Reward Modeling** | Predict text quality | Predict aesthetics, safety, faithfulness |
| **RL Fine-Tuning**  | PPO on text sequences | Requires pixel-space optimization |

---

## **2. How RLHF Works for Image Models**
### **Step 1: Supervised Fine-Tuning (SFT)**
- Train the base model (e.g., Stable Diffusion) on high-quality, curated images.
- Helps the model generate better initial outputs before alignment.

### **Step 2: Reward Model Training**
- Collect **human preference data** by showing users multiple generated images and asking:  
  - *Which image is more aesthetically pleasing?*  
  - *Which image better follows the prompt?*  
  - *Which image is safer (no harmful content)?*  
- Train a **reward model** (e.g., a neural network) to predict human preferences.

### **Step 3: RL Fine-Tuning (PPO or Diffusion Policy Optimization)**
- Use **Reinforcement Learning (RL)** to fine-tune the image generator to maximize the reward score.  
- Unlike text models, optimizing in **pixel space** is computationally expensive, so alternatives include:  
  - **Latent-space optimization** (e.g., fine-tuning Stable Diffusion’s latent space).  
  - **Denoising Diffusion Policy Optimization (DDPO)** (a variant of PPO for diffusion models).  

#### **Example: Improving Aesthetics with RLHF**
- A model like **DALL·E 3** may use RLHF to ensure:  
  - Generated images match prompts more accurately.  
  - Images are more visually appealing (better lighting, composition).  
  - Avoids generating harmful/NSFW content.  

---

## **3. How DPO Works for Image Models**
Since DPO eliminates the need for a separate reward model, it can be more efficient for image alignment.

### **Step 1: Collect Preference Data**
- Humans rank pairs of images **(A, B)** based on:  
  - **Prompt faithfulness** (does it match the text?).  
  - **Aesthetics** (which looks better?).  
  - **Safety** (which is less harmful?).  

### **Step 2: Direct Optimization**
- Instead of training a reward model, **DPO directly adjusts the image generator’s weights** to increase the likelihood of preferred images over dispreferred ones.  
- Works well for **diffusion models** since it avoids unstable RL training.  

#### **Example: Reducing Bias with DPO**
- If a model generates stereotypical images (e.g., "CEO" always as a man), DPO can:  
  - Downweight biased images in training.  
  - Upsample diverse, fairer generations.  

---

## **4. Other Post-Training Strategies for Image Models**
### **A. Rejection Sampling (Best-of-N Filtering)**
- Generate **multiple images**, then pick the **best one** using:  
  - A **safety classifier** (e.g., NSFW filter).  
  - A **reward model** (e.g., aesthetic scorer).  
- Used in **Midjourney v6** to improve output quality.  

### **B. Adversarial Training (Red-Teaming)**
- Test the model with **malicious prompts** (e.g., requests for violent images).  
- Fine-tune the model to **refuse harmful generations**.  

### **C. Human-in-the-Loop Refinement**
- Platforms like **Stable Diffusion XL** allow users to **rate images**, which are then used for further fine-tuning.  

---

## **5. Applications & Trade-offs**
| **Goal**               | **Method**          | **Pros**                          | **Cons**                          |
|-----------------------|--------------------|----------------------------------|----------------------------------|
| **Better Aesthetics**  | RLHF + Reward Model | High-quality outputs             | Computationally expensive        |
| **Prompt Faithfulness** | DPO               | Simpler, no reward model needed  | Needs large preference dataset   |
| **Safety/NSFW Filtering** | Rejection Sampling | Easy to implement               | Limited to filtering, not training |
| **Bias Mitigation**    | DPO + Fairness Data | Reduces stereotypes              | May reduce creativity            |

---

## **6. Real-World Examples**
- **OpenAI’s DALL·E 3** – Uses RLHF to improve prompt adherence and safety.  
- **Stable Diffusion XL** – Leverages human feedback for better aesthetics.  
- **Midjourney** – Uses ranking systems to refine style and quality.  

---

### **Conclusion**
- **RLHF and DPO can align image models** with human preferences, but **pixel-space optimization is harder than text**.  
- **DPO is more efficient** than RLHF for images but requires good preference data.  
- **Hybrid approaches (RLHF + filters)** are common in production systems.  

Would you like details on implementing this for a specific image model (e.g., Stable Diffusion)?
