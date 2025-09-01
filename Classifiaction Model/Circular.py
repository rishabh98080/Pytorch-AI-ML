# main.py
import torch
import torch.nn as nn # nn is the heart of PyTorch for building neural networks
import numpy as np    # We'll use numpy to create our sample data
import matplotlib.pyplot as plt # Import for plotting
from sklearn.datasets import make_circles # Import to create concentric circle data

# --- Step 1: Create Concentric Circle Data ---
# We'll use scikit-learn's make_circles to generate our data.
# This is the classic "non-linear" problem.
num_samples = 200
X_numpy, y_numpy = make_circles(n_samples=num_samples, noise=0.1, factor=0.5, random_state=42)

# --- Step 2: Convert data to PyTorch Tensors ---
# Same as before, we convert our numpy arrays to PyTorch tensors.
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1)

# --- Step 3: Define a Deeper Neural Network Model ---
# To solve a non-linear problem, we need a more powerful model.
# Let's add another hidden layer to make it "deeper".

class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        # The __init__ method is where you define the layers of your network.
        super(BinaryClassificationModel, self).__init__()

        # Layer 1
        self.layer_1 = nn.Linear(input_size, hidden_size_1)
        self.relu_1 = nn.ReLU()

        # Layer 2 (The new layer!)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu_2 = nn.ReLU()

        # Output Layer
        self.layer_3 = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        # The "forward" method defines how data flows through the network.
        
        # Pass through layer 1
        out = self.layer_1(x)
        out = self.relu_1(out)
        
        # Pass through layer 2
        out = self.layer_2(out)
        out = self.relu_2(out)
        
        # Pass through output layer
        out = self.layer_3(out)
        return out

# --- Step 4: Create the Model, Loss Function, and Optimizer ---

# Create an instance of our NEW, deeper model.
# We'll use more neurons in our hidden layers.
model = BinaryClassificationModel(input_size=2, hidden_size_1=16, hidden_size_2=16)

# Loss Function and Optimizer remain the same.
criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Step 5: The Training Loop ---
# This problem is harder, so we need more training epochs for the model to learn.
num_epochs = 1000 

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# --- Step 6: Evaluate the Model ---
with torch.no_grad():
    predictions_raw = model(X)
    predictions_prob = torch.sigmoid(predictions_raw)
    predictions_class = predictions_prob.round()
    accuracy = (predictions_class.eq(y).sum() / float(y.shape[0]))
    print(f'\nTraining complete!')
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')

# --- Step 7: Visualize the Results ---
print('\nGenerating plot...')
with torch.no_grad():
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    Z = model(grid_tensor)
    Z = torch.sigmoid(Z).reshape(xx.shape).round()

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X_numpy[:, 0], X_numpy[:, 1], c=y_numpy, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Concentric Circles with Non-Linear Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    plt.savefig('classification_plot.png')
    print("Plot saved to classification_plot.png")
