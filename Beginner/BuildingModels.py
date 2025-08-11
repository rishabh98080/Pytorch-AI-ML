import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from SavingModel import SavingModels


weight = 0.7
bais = 0.3

start  = 0
end = 1

step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim = 1)
Y = weight * X + bais

#print(X[:10],"\n",Y[:10])
#print(len(X),"\n",len(Y))
train_split = int(0.8 * len(X))
x_train,y_train = X[:train_split], Y[:train_split]
x_test,y_test = X[train_split:], Y[train_split:]





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def plot_graph(train_data = x_train,
               train_label = y_train,
               test_data = x_test,
               test_label = y_test,
               prediction = None):
    plt.figure(figsize=(8,6))
    plt.scatter(train_data, train_label, color='blue', label='Train Data')
    plt.scatter(test_data, test_label, color='orange', label='Test Data')
    if prediction is not None:
        plt.scatter(test_data, prediction, color='red', label='Predictions')
    plt.legend()
    plt.show()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$







###############################################################################################################

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                               requires_grad= True,
                                               dtype = torch.float))
        self.biases = nn.Parameter(torch.randn(1,
                                            requires_grad= True,
                                            dtype = torch.float))        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.biases

#######################################################################################################################




torch.manual_seed(42)
model1 = LinearRegressionModel()
with torch.inference_mode():
    y_preds = model1.forward(x_test)
print(y_test[:10])





##############################################################################



loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model1.parameters(), lr=0.001)

epochs = 2000

for epoch in range(epochs):
    model1.train()
    y_preds = model1(x_train)
    loss = loss_fn(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model1.eval()
    with torch.inference_mode():
        y_preds_test = model1(x_test)
        print(list(model1.parameters()))



#*******************************************************************************************************




# After training, evaluate on test set


plot_graph(x_train, y_train, x_test, y_test, y_preds_test)
SaveModel = SavingModels(model1)
print(SaveModel)