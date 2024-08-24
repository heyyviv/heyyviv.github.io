---
layout: post
title: " Normalisation "
date: 2024-08-24 01:43:18 +0530
categories: Deep Learning
---

Different Normalization Layers in Deep Learning

Batch Normalization(BN)

Batch Normalization focuses on standardizing the inputs to any particular layer(i.e. activations from previous layers). Standardizing the inputs mean that inputs to any layer in the network should have approximately zero mean and unit variance. Mathematically, BN layer transforms each input in the current mini-batch by subtracting the input mean in the current mini-batch and dividing it by the standard deviation.

But each layer doesn’t need to expect inputs with zero mean and unit variance, but instead, probably the model might perform better with some other mean and variance. Hence the BN layer also introduces two learnable parameters γ and β.

The whole layer operation is as follows. It takes an input x_i and transforms it into y_i as described in the below table.

![BN](/assets/norn_1.jpg)

