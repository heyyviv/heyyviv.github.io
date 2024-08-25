---
layout: post
title: "Inception"
date: 2024-08-24 01:43:18 +0530
categories: Deep Learning
---

Going deeper with convolutions

Sparse matrices: This term refers to any matrix that is mainly composed of zeros. (In contrast, dense matrices are mostly composed of non-zero values). Matrix is a matrix, of course, but by dealing with sparse matrices as a special data structure, we can save computational resources. In the notoriously resource-hungry world of machine learning, its importance cannot be overstated. Sparse matrices can be saved as dictionaries of keys, lists of lists, or more specialized structures, such as compressed sparse rows.

1x1 convolution: This layer is added to the neural network to keep the computational cost maintainable. Normally, the matrix is downsampled (through max-pooling) to achieve this, however, we cannot perform downsampling too many times, otherwise, information loss accumulates, which we want to avoid. 1x1 convolution solves this problem by pooling not inside a single channel, but across the channels themselves (Figure 1).

A large kernel size is used to capture a global distribution of the image while a small kernel size is used to capture more local information.

Inception network architecture makes it possible to use filters of multiple sizes without increasing the depth of the network. The different filters are added parallelly instead of being fully connected one after the other.

![Inception](/assets/inception_1.jpg)
This is known as the naive version of the inception model. The problem with this model was the huge number of parameters. To mitigate the same, they came up with the below architecture.
![Inception](/assets/inception_2.jpg)

How does this architecture reduce dimensionality?
Adding a 1X1 convolution before a 5X5 convolution would reduce the number of channels of the image when it is provided as an input to the 5X5 convolution, in turn reducing the number of parameters and the computational requirement.

![Inception](/assets/inception_3.jpg)

![Inception](/assets/inception_4.jpg)

What is different in the Inception V3 network from the inception V1 network?

Inception V3 is an extension of the V1 module, it uses techniques like factorizing larger convolutions to smaller convolutions (say a 5X5 convolution is factorized into two 3X3 convolutions) and asymmetric factorizations (example: factorizing a 3X3 filter into a 1X3 and 3X1 filter).
These factorizations are done with the aim of reducing the number of parameters being used at every inception module. Below is an image of the inception V3 module.

![Inception](/assets/inception_5.jpg)



