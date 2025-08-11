import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#scalar = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
#print(scalar.ndim)
#scalar.shape


#myArray = np.array([-1,0,1])
#print(myArray.size)

#print(myArray.dtype)

#newArry = myArray.astype('bool')
#print(newArry)

#random_number =  torch.rand(20,5,6)
#print(random_number , random_number.ndim)

#turning images in tensors

##image_tensor = torch.rand(1080,720,3) #width,height,colors(rgb)
#print(image_tensor)

### zeroes and ones tensors

#zeros = torch.zeros((3,5))
#ones = torch.ones((5,6)) # dtype stands for default type
#print(ones,ones.dtype)

#letArray = torch.arange(2,9,2) # .range() return float while arange return int
#print(letArray)
           
#ten_zeros = torch.zeros_like(letArray)
#print(ten_zeros)
#tens_ones = torch.ones_like(letArray)
#print(tens_ones) # 1: 59 : 23

#float_32_tensor  = torch.tensor([3.0,6.0,9.0],dtype = torch.float16 , device = "cpu" , requires_grad = False)
#print(float_32_tensor,float_32_tensor,float_32_tensor.dtype,float_32_tensor.device,float_32_tensor.requires_grad)
#import torch

 # Number of available CUDA devices
#print(torch.get_default_device)  # Name of the first GPU

#Tensor = torch.tensor([1,2,3,5,6,6,6],device ="cuda:0",dtype = int)
#Tensor1 = torch.tensor([3,43,34,2,4,2,2],dtype= int,device ="cuda:0")

#print(Tensor + Tensor1)

#torch.add(tensor,10) # add 10 to each element of tensor
#torch.mul(Tensor,Tensor1) # add two tensors element wise
#matrx = torch.rand(2,3)
#print(matrx)
#matrex = torch.rand(3,2)
#print(matrex)

#print(torch.matmul(matrex,matrx)) # matrix multiplication
mat1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
mat2 = torch.tensor([[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]])

if(mat1.shape == mat2.shape):
    print(torch.matmul(mat1,mat2.T).type(torch.float32)) # matrix multiplication with transpose

print(torch.max(mat1,dim = 1),mat1.max()) # dim = 0,1 0 = column, 1 = row

print(torch.sum(mat1[1] + mat1[2]))
print(mat1[1],mat1[2])

print(torch.mean(mat1,dtype=torch.float32,dim = 1)) # mean of all elements

marks = torch.tensor([1,2,3,4,5,6,7,8,9,10])

print(marks[torch.argmin(marks).item()].item())

how = torch.tensor([1,2,3,4,5,6,7,8,9,10])
print(how)