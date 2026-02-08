import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = x ** 2 + y ** 3
z.backward()

print(f"Gradient of x: {x.grad}")  # 2 * x = 4
print(f"Gradient of y: {y.grad}")  # 3 * y^2 = 27