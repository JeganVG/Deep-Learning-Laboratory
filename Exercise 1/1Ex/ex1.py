import torch
x = torch.randn(4,3)
print(x)
tensor = torch.ones(4, 4)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")