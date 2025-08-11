import torch
import numpy as np

seq =  torch.arange(0.0,100.0)
array = np.array([[float(i)]for i in  range(200,300)])  # Creating a numpy array with a condition
#print(array,array.shape,array.ndim)
newseq = torch.tensor(array)


#print(seq.shape)
#print(seq.reshape(20,5))
#print(seq.view(25,4))
#print(torch.vstack([seq,newseq]),torch.vstack([seq,seq,seq]).ndim)  # Stacking along a new dimension

#print(torch.hstack([seq,newseq]),torch.hstack([seq, seq, seq]).ndim)  # Stacking along the existing dimension


#print(torch.squeeze(newseq,dim = 0 ))  # Adding a new dimension at the front

newArray = torch.rand(size = (2,3,5,6))
print(newArray.shape,newArray.size(),newArray.ndim)  # Getting the shape, size, and number of dimensions of a tensor


print(newArray,'\n',newArray.permute(1,3,2,0))