import torch
import numpy as bp
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

random_seed = 42
torch.manual_seed(random_seed)

x = torch.rand((3,4), device=device )

torch.manual_seed(random_seed)
y = torch.rand(size= (3,4), device=device)

print(x == y)