import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.arange(1, 21).reshape(1, 1, 4, 5)
print(x)

x_np = x[0, 0, :, :].numpy()

plt.imshow(x_np, cmap='viridis')  # You can choose different colormaps
plt.colorbar()  # Add a colorbar to show the mapping of values to colors
plt.title("Visualization of the 4x5 slice of the tensor")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.xticks(np.arange(x_np.shape[1]))
plt.yticks(np.arange(x_np.shape[0]))
plt.grid(True)
plt.show()