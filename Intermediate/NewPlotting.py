import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np # Often needed for plotting
import seaborn as sns # For pretty plots like the confusion matrix
from sklearn.metrics import confusion_matrix

#======================================================================#
#                                                                      #
#                YOUR ORIGINAL CODE (UNCHANGED)                        #
#                                                                      #
#======================================================================#

# 1. Create the data
n_samples = 1000
X, Y = make_circles(n_samples=n_samples, noise=0.05, random_state=42)

# 2. Convert to tensors
X = torch.from_numpy(X).type(torch.float32)
Y = torch.from_numpy(Y).type(torch.float32)

# 3. Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=42)

# 4. Build the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Note: Your model is a linear model because it doesn't have non-linear activation
# functions between the layers. It will struggle to fit the non-linear circle data!
# This is something our new plots will make very clear.
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)


# 5. Setup Loss function and Optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1) # Increased lr for faster training

# 6. Training Loop
torch.manual_seed(42)
epochs = 1000 # Reduced epochs for quicker demonstration

X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)

for epoch in range(epochs):
    ## Training Mode
    model_0.train()
    
    # Forward pass (logits)
    y_logits = model_0(X_train).squeeze()
    
    # Calculate Loss
    loss = loss_fn(y_logits, Y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Gradient descent
    optimizer.step()

    ### Testing (optional, but good practice)
    if epoch % 100 == 0:
        model_0.eval()
        with torch.inference_mode():
            test_logits = model_0(X_test).squeeze()
            test_loss = loss_fn(test_logits, Y_test)
        # print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f}")


#======================================================================#
#                                                                      #
#           NEW PLOTTING TECHNIQUES START HERE                         #
#                                                                      #
#======================================================================#


# First, let's get our model's final predictions on the test set
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test)

# The model outputs raw "logits". To get actual 0 or 1 predictions,
# we need to pass them through a sigmoid function and then round them.
y_preds = torch.round(torch.sigmoid(y_logits)).squeeze()


# --- Technique 1: A Better Scatter Plot (Correct vs. Incorrect) ---
# This plot shows where the model got it right and where it went wrong.
print("\n--- Plot 1: Correct vs. Incorrect Predictions ---")
plt.figure(figsize=(10, 7))
plt.title("Model Predictions (Correct vs. Incorrect)")
# Plot the points where the model was correct
plt.scatter(X_test[y_preds == Y_test][:, 0].cpu(),
            X_test[y_preds == Y_test][:, 1].cpu(),
            c='g', marker='o', s=30, label='Correct')
# Plot the points where the model was wrong
plt.scatter(X_test[y_preds != Y_test][:, 0].cpu(),
            X_test[y_preds != Y_test][:, 1].cpu(),
            c='r', marker='x', s=60, label='Incorrect')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


# --- Technique 2: The Decision Boundary Plot ---
# This is the most powerful visualization for classification. It shows the
# background colored by what the model would predict for any point in that area.
# It literally draws the line the model has learned.
print("\n--- Plot 2: Decision Boundary ---")

def plot_decision_boundary(model, X, y):
    # Put everything to CPU for plotting
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)
    
    y_pred = torch.round(torch.sigmoid(y_logits)).reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Let's call our new function!
plot_decision_boundary(model_0, X_test, Y_test)


# --- Technique 3: The Confusion Matrix ---
# This is a table that gives you a much clearer picture of performance
# than a single accuracy score. It shows what kind of errors are being made.
print("\n--- Plot 3: Confusion Matrix ---")
conf_matrix = confusion_matrix(Y_test.cpu(), y_preds.cpu())

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, 
            annot=True, # Show the numbers in the squares
            fmt='g',    # Use plain number format
            cmap='Blues',
            xticklabels=['Predicted Class 0', 'Predicted Class 1'],
            yticklabels=['Actual Class 0', 'Actual Class 1'])
plt.title("Confusion Matrix")
plt.show()