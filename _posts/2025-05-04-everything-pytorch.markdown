---
layout: post
title: "Everything PyTorch"
date: 2025-05-04 01:43:18 +0530
categories: deep_learning
---

PyTorch feels like NumPy, but with GPU acceleration and automatic computation of gradients, which makes it suitable for calculating backward pass data automatically starting from a forward expression.

Best feature about pytorch is :
- PyTorch provides allows tensors to keep track of the operations performed on them and to compute derivatives of an output with respect to any of its inputs analytically via backpropagation. This capability is provided natively by tensors and further refined in torch.autograd

PyTorch defaults to an immediate execution model (eager mode). Whenever an instruction involving PyTorch is executed by the Python interpreter, the corresponding operation is immediately carried out by the underlying C++ or CUDA implementation. As more instructions operate on tensors, more operations are executed by the backend implementation. This process is as fast as it typically can be on the C++ side, but it incurs the cost of calling that implementation through Python. This cost is minute, but it adds up.

# Tensor
Tensor is array that can access by an index or set of indices.
Many types of data—from images to time series, audio, and even sentences—can be represented by tensors.

```python 
import torch
a = torch.ones(3)
a
o/p:
tensor([1., 1., 1.]) 
```
PyTorch tensors or NumPy arrays,  are views over (typically) contiguous memory blocks containing unboxed C numeric types, not Python objects. 
Python lists or tuples of numbers are collections of Python objects that are individually allocated in memory.

![Tensor](/assets/pytorch_1.png)
The dtype argument to tensor constructors (that is, functions such as tensor, zeros, and ones) specifies the numerical data type that will be contained in the tensor.

![Tensor](/assets/pytorch_2.jpg)

PyTorch uses pickle under the hood to serialize the tensor object, as well as dedicated serialization code for the storage. 

save: 
torch.save(points, '../data/p1ch3/ourpoints.t')
or
with open('../data/p1ch3/ourpoints.t','wb') as f:
    torch.save(points, f)

Loading your points:
points = torch.load('../data/p1ch3/ourpoints.t')
or
with open('../data/p1ch3/ourpoints.t','rb') as f:
    points = torch.load(f)

In addition to the dtype, a PyTorch tensor has a notion of device, which is where on the computer the tensor data is being placed.
points_gpu = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 4.0]],device='cuda')
You could instead copy a tensor created on the CPU to the GPU by using the to method:
points_gpu = points.to(device='cuda')

Tabular data is heterogenous it contain date,category,time,number etc.
Your first job as a deep learning practitioner, therefore, is to encode heterogenous, real-world data in a tensor of floating-point numbers, ready for consumption by a neural network.

data = wineq[:, :-1] 
select all rows except the last column.
target = wineq[:, -1]
select all rows and only last column.

attributes of a tensor:
```python
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```
By default, tensors are created on the CPU. We need to explicitly move tensors to the accelerator using .to method (after checking for accelerator availability). Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!

Joining tensors You can use torch.cat to concatenate a sequence of tensors along a given dimension. 
```python
tensor = torch.ones(4, 4)
torch.cat([tensor,tensor,tensor],dim=1)
torch.cat([tensor,tensor,tensor],dim=0)
```
op:
1.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
2.tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])

matrix multiplication:  (tensor.T returns the transpose of a tensor)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
element wise multiplication
z1 = tensor * tensor
z2 = tensor.mul(tensor)
Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

# Datasets & DataLoaders
Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

custom dataset:
```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```
The __init__ function is run once when instantiating the Dataset object. We initialize the directory containing the images, the annotations file, and both transforms.
The __len__ function returns the number of samples in our dataset.
The __getitem__ function loads and returns a sample from the dataset at the given index idx. Based on the index, it identifies the image’s location on disk, converts that to a tensor using read_image, retrieves the corresponding label from the csv data in self.img_labels, calls the transform functions on them (if applicable), and returns the tensor image and corresponding label in a tuple.

The Dataset retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.

DataLoader is an iterable that abstracts this complexity for us in an easy API.
```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

# Build the Neural Network
The torch.nn namespace provides all the building blocks you need to build your own neural network.
```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```
We create an instance of NeuralNetwork, and move it to the device, and print its structure.
model = NeuralNetwork().to(device)
print(model)
out: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined.
Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training. Subclassing nn.Module automatically tracks all fields defined inside your model object, and makes all parameters accessible using your model’s parameters() or named_parameters() methods.

# Automatic Differentiation
PyTorch has a built-in differentiation engine called torch.autograd. It supports automatic computation of gradient for any computational graph.

```python
import torch
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```
![Gradient](/assets/pytorch_3.jpg)

By default, all tensors with requires_grad=True are tracking their computational history and support gradient computation. However, there are some cases when we do not need to do that, for example, when we have trained the model and just want to apply it to some input data, i.e. we only want to do forward computations through the network. We can stop tracking computations by surrounding our computation code with torch.no_grad() block:
```python
with torch.no_grad():
    z = torch.matmul(x, w)+b
```
There are reasons you might want to disable gradient tracking:
- To mark some parameters in your neural network as frozen parameters.
- To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.
