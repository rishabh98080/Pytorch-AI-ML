import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import numpy as np # Import numpy for the visualization function
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split


# Your original data generation and preparation
n_samples = 1000
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
    nn.Linear(in_features=5,out_features= 1)
)

# Your original loss function and optimizer, unchanged
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(),lr =0.01)


#################################################################
## WARNING: The following training loop is from your original code.
## It will print the model's weights 10,000 times,
## which will result in a VERY large amount of text output.
#################################################################
print("\nStarting training... (This will produce a lot of output)")

# Your original training loop, unchanged
torch.manual_seed(42)
epochs = 10000

for epoch in range(epochs):
    ## Training Mode
    model_0.train()
    y_logits = model_0(X_train)
    # y_prediction = torch.round(torch.sigmoid(y_logits)).squeeze(dim = 1) # This line was not used in your loop
    
    ### Loss calc
    Loss = loss_fn(y_logits.squeeze(),Y_train)

    optimizer.zero_grad()

    Loss.backward()

    optimizer.step()

    # This is the original print statement from your code
    if epoch < 5 or epoch > epochs - 5: # Let's just print the first and last few to avoid crashing
        print(f"Epoch {epoch}:")
        print(model_0.state_dict())
        print("-" * 20)

print("Training finished.")

#################################################################
## NEW VISUALIZATION PART
## This is the only part that is different.
## We are using the advanced plotting function to see what your
## original linear model learned.
#################################################################
print("\nGenerating visualization...")

# The helper function for visualization
def plot_decision_boundary(model, X, y):
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))
    X_to_pred_on = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)
    y_pred = torch.round(torch.sigmoid(y_logits)).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Creating the plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train Data - Linear Model")
plot_decision_boundary(model_0, X_train, Y_train)
plt.subplot(1, 2, 2)
plt.title("Test Data - Linear Model")
plot_decision_boundary(model_0, X_test, Y_test)
plt.show()

print("Done!")