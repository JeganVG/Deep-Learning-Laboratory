
# Simple Autograd Implementation
import torch
from torch import *
from torch.autograd import Variable
mat1 = Variable(torch.Tensor([[5,4],[7,6]]),requires_grad=True)
print(mat1)
mat2 = Variable(torch.Tensor([[3,1],[7,9]]),requires_grad=True)
print(mat2)

output_pred = torch.sum(mat1**mat2)
print(output_pred)
output_pred.backward()

x = torch.tensor(7., requires_grad = False)
w = torch.tensor(5., requires_grad = True)
b = torch.tensor(2., requires_grad = True)
tensor(1., requires_grad=True)
print(x)
print(w)
print(b)
y = w**2 + x**b
print(y)
y.backward()