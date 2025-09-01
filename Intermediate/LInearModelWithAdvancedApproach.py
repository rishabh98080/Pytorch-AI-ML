import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

weight = 0.2
bias = 0.6
X = torch.randn(100,device = device).unsqueeze(dim = 1)
Y = X * weight + bias

x_train,y_train =  X[:51],Y[:51]
x_test,y_test = X[51:100],Y[51:100]


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features = 1, out_features = 2)
        self.layer_2 = nn.Linear(in_features= 2, out_features = 1)

    def forward(self,X):
        return self.layer_2(self.layer_1(X)) 
    
torch.manual_seed(42)

model_0 = LinearModel()


loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params= model_0.parameters(),lr = 0.01)


epochs= 10

for epoch in range(epochs):
    model_0.train()
    y_prediction = model_0(x_train)
    Loss = loss_fn(y_prediction,y_train)

    optimizer.zero_grad()

    Loss.backward()

    optimizer.step()

    print(f"Loss : {Loss}, {model_0.state_dict()}")


with torch.inference_mode():
    model_0.eval()
    y_test_predictions = model_0(x_test)
    print(loss_fn(y_test_predictions,y_test))


###############################################################################
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
################################################################################

plot_graph(prediction = y_test_predictions)