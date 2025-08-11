import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


########### DEVICE AGNOSTICISM ###############

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using....{device}")


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



wt = 2.5
bias = 3.6

my_dataset_X = torch.arange(3,6,0.134568).unsqueeze(dim = 1)
my_dataset_Y = wt * my_dataset_X + bias

print(my_dataset_X , " \n" , my_dataset_Y)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



########    DATA SPLIT   #####################

x_train = my_dataset_X[:15]
y_train = my_dataset_Y[:15]


x_test = my_dataset_X[15:23]
y_test = my_dataset_Y[15:23]



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#### Plot Function ##############

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_graph(x_train,y_train,x_test,y_test,y_preds = None):
    plt.figure(figsize=(8,8))
    plt.scatter(x_train,y_train,color = 'blue',marker = 'o',label = 'Train_Data')
    plt.scatter(x_test,y_test,color = 'green',marker = 'o',label = 'Test_Data')
    if y_preds is not None:
        plt.scatter(x_test,y_preds,color = 'red',marker = 'o',label = 'Predition_Data')
    plt.legend()
    plt.show()

plot_graph(x_train,y_train,x_test,y_test,y_preds=None)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


########### MODEL ###########
torch.manual_seed(42)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.weight = nn.Parameter(torch.randn(1,dtype = torch.float,requires_grad= True))  ### TRY USING NN.Linear()
        #self.biases = nn.Parameter(torch.randn(1,dtype=torch.float,requires_grad=True))
        self.linear_layer = nn.Linear(in_features=1,out_features=1)
    def forward(self,x):
        return self.linear_layer(x)

#*****************************************************************************************************
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


model_0 = LinearRegressionModel()

#with torch.inference_mode():
 #   y_preds = model_0.forward(x_test)
#print("Prediction_Size : " , y_preds.size())

#plot_graph(x_train,y_train,x_test,y_test,np.array(torch.tensor(y_preds)))

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

#################### LOSS AND OPTIMIZATION #######################


loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params= model_0.parameters(),lr = 0.01)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

epochs = 1

for epoch in range(epochs):
    print(f"iterations : {epoch}")
    model_0.train()
    y_preds = model_0(x_train)
    loss = loss_fn(y_preds, y_train)
    optimizer.zero_grad()   ## sets the gradient from previous iteration to zero 
    loss.backward()  ## it calculate the gradient using loss through backpropagation
    optimizer.step()  ## it applies the new gradient calculated
    with torch.inference_mode():
        model_0.eval()
        y_preds_test = model_0(x_test)
        print(list(model_0.parameters()))


print("x_y_train_size : ", x_train.size() , " ",y_train.size()," x_y_test : ",x_test.size()," ",y_test.size() , " preds_size : " , torch.tensor(y_preds).size())
plot_graph(x_train,y_train,x_test,y_test,y_preds_test)



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


######              SAVE AND LOAD MODEL                      ##########################



######################## SAVE THE MODEL ############################

import os

Model_Path = './model'

if not os.path.exists(Model_Path):
    os.mkdir(Model_Path)
    torch.save(model_0.state_dict(),Model_Path + "/model_1.pth")

else:
    torch.save(model_0.state_dict(),Model_Path + "/model_1.pth")

#####################  LOAD THE MODEL #############

