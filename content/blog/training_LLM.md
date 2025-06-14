+++
title = "Training_LLM"
date = "2025-05-23T15:12:55+05:30"

# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["training","LLM","notes"]
+++

# Training on One GPU

when a model trained, there are 3 phases
- A forward pass, which passes inputs through the model to yield its outputs
- A backward pass to compute the gradients
- An optimization step using the gradients to update the parameters
The batch size (bs) is one of the important hyperparameters for model training; it affects both model convergence and throughput.

A small batch size can be useful early in training to quickly move through the training landscape to reach an optimal learning point. However, further along in the model training, small batch sizes will keep gradients noisy, and the model may not be able to converge to the most optimal final performance. At the other extreme, a large batch size, while giving very accurate gradient estimations, will tend to make less use of each training token, rendering convergence slower and potentially wasting compute resources.

Batch size also affects the time it takes to train on a given text dataset: a small batch size will require more optimizer steps to train on the same amount of samples. Optimizer steps are costly (in compute time), and the total time to train will thus increase compared to using a larger batch size. That being said, note that the batch size can often be adjusted quite widely around the optimal batch size without major impact on the performance of the model - that is, the sensitivity of final model performance to the exact batch size value is usually rather low around the optimal batch size.
In the LLM pretraining community, batch sizes are commonly reported in terms of tokens rather than number of samples
bst = batch size tokens
bs = batch size
seq = model input sequence length
bst = bs * seq
Llama 1 was trained with a batch size of ~4M tokens for 1.4 trillion tokens, while DeepSeek was trained with a batch size of ~60M tokens for 14 trillion tokens.

we couldn't calculate exact memory usage by a model cuz
- CUDA kernels typically require 1-2 GB of GPU memory
- Some memory is used for buffers and intermediate results, and there's some memory that can't be used due to fragmentation.
We could face out-of-memory (OOM) issues when training this large models but why?
When training a neural network model, we store several items in memory:
- Model weights
- Model gradients
- Optimizer states
- Activations needed to compute the gradients

First the activations increase quickly as we do the forward pass, then during the backward pass the gradients build up, and as the backward pass propagates, the stored activations used to compute the gradients are progressively cleared. Finally, we perform optimization, during which we need all the gradients, and then update the optimizer states before we start the next forward pass.

An interesting observation here is that memory usage is not static for a given model; rather, it scales linearly with the batch size and quadratically with the sequence length. This means the activation memory is the part that will blow up when we increase our batch size or train with longer sequences. 

These graphs tell a striking story: for short sequences (or small batch sizes), memory usage for activations is almost negligible, but from around 2-4k tokens they start to take up a significant amount of memory, while usage for parameters, gradients, and optimizer states (as we’ll discuss later) is roughly independent of the sequence length and batch size.
The general idea behind activation recomputation – also called gradient checkpointing or rematerialization – is to discard some activations during the forward pass to save memory and spend some extra compute to recompute these on the fly during the backward pass. Without recomputation, we store every hidden state between two learnable operations (e.g., feedforward, LayerNorm, etc.), so that we can use them during the backward pass to compute gradients. When we use recomputation, we typically only store activations at a few key points in the model architecture, discarding the rest of the activations and recomputing them on the fly during the backward pass from the nearest saved activations. Basically, we perform a sub-part of the forward pass again, to trade off memory for compute. 

- FULL : We checkpoint activations at the transition point between each layer of the Transformer model. This is usually called the “full” strategy since it requires a forward pass through each layer, essentially adding a full forward pass during the backward pass. This strategy saves the most memory but is the most expensive one in terms of compute. It typically increases the compute cost and time by up to 30-40%, which is very noticeable.
- Selective: In general, we can do better than full. The authors of the recomputation paper did a detailed analysis studying which activations grow the largest and have the cheapest recomputation cost in terms of floating-point operations per second (FLOPS). It turns out that the attention computations fall in that category, and thus we can usually discard them and focus on checkpointing the expensive feedforward computations. For a GPT-3 (175B) model, this means a 70% activation memory reduction at a 2.7% compute cost.

Gradient accumulation is a very straightforward method to avoid memory explosion that consists of splitting a batch into micro-batches. We then perform forward and backward passes successively on each micro-batch, compute the gradients, and, as the name suggests, sum the gradients of all micro-batches before we perform optimization. In practice, the optimization step is conducted not on the sum but on the average of the gradients, so that the result is independent of the number of gradient accumulation steps.
Gradient accumulation allows us to reduce activation memory, which grows linearly with batch size, by processing smaller micro-batches sequentially. This reduces stored activations and gradients since only one micro-batch's worth of activations needs to be kept in memory at a time, which helps reduce the overall activation memory footprint.
One drawback, however, is that gradient accumulation requires multiple consecutive forward/backward passes per optimization step, thereby increasing the compute overhead and slowing down training. 

# Data Parallelism
The idea behind data parallelism (DP) is to replicate the model on several GPUs (we call the replicas “model instances”) and run forward and backward passes on different micro-batches of data in parallel on each GPU - hence the name data parallelism. 
Using a different micro-batch for each GPU means we’ll have different gradients on each GPU, so to keep the model instances in sync across the different GPUs, we'll average the gradients from the model instances using an operation called “all-reduce.” This operation takes place during the backward pass, before the optimizer step.
