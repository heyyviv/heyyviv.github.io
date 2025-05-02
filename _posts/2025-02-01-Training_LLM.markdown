---
layout: post
title: "Training LLM"
date: 2024-10-16 01:43:18 +0530
categories: LLM
---


# Why do we need to optimize training 
- Memory Usage: it's a hard limitation - if a training step doesn't fit in memory, training cannot proceed
- Compute Efficiency: we want our hardware to spend most time computing, so we need to reduce time spent on data transfers or waiting for other GPUs to perform work.
- Communication overhead: we want to minimize communication overhead as it keeps GPUs idle. To archieve this we will try to make best use of intra-node (fast) and inter-node (slower) bandwidths as well as overlap communication with compute as much as possible.

![LLM](/assets/gpu_train.jpg)
![LLM](/assets/gpu_train1.jpg)
![LLM](/assets/gpu_train2.jpg)

# First Steps: Training on one GPU
When a model is trained on a single GPU, the training typically consists of three steps:
- a forward pass which passes inputs through the model to yield its outputs,
- a backward pass to compute the gradients, and
- an optimization step using the gradients to update the parameters

![LLM](/assets/gpu_train3.jpg)

The batch size (bs) is one of the important hyper-parameters for model training and affects both model convergence and throughput.
A small batch size can be useful early in training to quickly move along the training landscape reaching an optimal learning point. However, further along the model training, small batch sizes will keep gradients noisy and the model may not be able to converge to the most optimal final performances. At the other extreme, a large batch size while giving very accurate gradient estimations will tend to make less use of each training token rendering convergence slower and potentially wasting compute. 
