import torch
import numpy as np
deivce = 'cuda' if torch.cuda.is_available() else 'cpu'

array = np.arange(1,20)

tensor = torch.tensor(array,dtype = torch.float32)

print(array.dtype,tensor.dtype)

narray = np.array(tensor)
print(narray.dtype)
