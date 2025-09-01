import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import numpy as np # Import numpy for the visualization function
import torch
from torch import nn
from torch import optim
import pandas as pd
from sklearn.model_selection import train_test_split


# Your original data generation and preparation
n_samples = 10000
X,Y = make_circles(n_samples=n_samples, noise=0.05,random_state=42)
print(X.shape)
print(Y.shape)
circles = pd.DataFrame(X,columns=['x1','x2'])
circles['label'] = Y
print(circles.head(10))
# plt.scatter(circles['x1'],circles['x2'],c=circles['label'],cmap=plt.cm.RdYlBu)
# plt.show()


# Your original tensor conversion and split
X = torch.from_numpy(X).type(torch.float32)
Y = torch.from_numpy(Y).type(torch.float32)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state = 42)

# Your original model definition, unchanged
# This is a linear model because it has no non-linear activation function.
torch.manual_seed(42)
model_0 = nn.Sequential(
    nn.Linear(in_features=2,out_features=5),
    nn.Linear(in_features=5,out_features=5),
    nn.Linear(in_features=5,out_features=5),
    nn.Linear(in_features=5,out_features=5),
    nn.Linear(in_features=5,out_features=5),
    nn.Linear(in_features=5,out_features= 1)
)

# Your original loss function and optimizer, unchanged
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(),lr =0.001)

def accuracy(y_true,y_predictions):
    correct = torch.eq(y_true,y_predictions).sum().item()
    acc = (correct/len(y_true)) * 100
    return acc

#################################################################
## WARNING: The following training loop is from your original code.
## It will print the model's weights 10,000 times,
## which will result in a VERY large amount of text output.
#################################################################
print("\nStarting training... (This will produce a lot of output)")

# Your original training loop, unchanged
torch.manual_seed(42)
epochs = 10

for epoch in range(epochs):
    ## Training Mode
    model_0.train()
    y_logits = model_0(X_train)
    y_prediction = torch.round(torch.sigmoid(y_logits)).squeeze(dim = 1) # This line was not used in your loop
    
    ### Loss calc
    Loss = loss_fn(y_logits.squeeze(),Y_train)

    Acc = accuracy(Y_train,y_prediction)

    optimizer.zero_grad()

    Loss.backward()

    optimizer.step()

    # This is the original print statement from your code
    if epoch < 5 or epoch > epochs - 5: # Let's just print the first and last few to avoid crashing
        print(f"Epoch {epoch}:")
        print(model_0.state_dict())
        print("-" * 20)
#print("Training finished.")
    model_0.eval()
    with torch.inference_mode():
        y_logits = model_0(X_test)
    y_pred = torch.round(torch.sigmoid(y_logits)).squeeze(dim = 1)

    test_loss = loss_fn(y_logits.squeeze(),Y_test)
    test_acc = accuracy(Y_test,y_pred)

    if epoch % 100 == 0:
        print(f"epoch:{epoch},Loss:{Loss:.4f},Accuracy:{Acc:.2f}% | test_epoch:{epoch},Test_Loss:{test_loss:.4f};Test_Accuracy:{test_acc:.2f}%")
print("Evaluation finished.")
