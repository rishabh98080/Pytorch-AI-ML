# main.py
import torch
import torch.nn as nn # nn is the heart of PyTorch for building neural networks
import numpy as np    # We'll use numpy to create our sample data
import matplotlib.pyplot as plt # Import for plotting

# --- Step 1: Create some dummy data ---
# We need data to train our model on. Let's create some simple data where
# the goal is to classify points into two groups (0 or 1).
# We'll create 100 data points, each with 2 features (like an x and y coordinate).

# np.random.seed(1) # Uncomment this line to get the same random data every time
num_samples = 100
num_features = 2

# Create the features (our X values).
# This creates a numpy array of shape (100, 2) with random numbers.
X_numpy = np.random.rand(num_samples, num_features)

# Create the labels (our y values).
# Let's say if the sum of the two features is less than 1.0, the class is 0.
# Otherwise, the class is 1. This gives our model a simple pattern to learn.
y_numpy = (np.sum(X_numpy, axis=1) < 1.0).astype(int)

# --- Step 2: Convert data to PyTorch Tensors ---
# PyTorch works with its own data structure called a "Tensor". It's very
# similar to a numpy array but has special properties for deep learning (like
# working with GPUs and calculating gradients).

# We convert our numpy arrays to PyTorch tensors.
# .float() makes them 32-bit floating point numbers, which is standard.
X = torch.from_numpy(X_numpy.astype(np.float32))

# The labels also need to be tensors. We use .view(-1, 1) to make sure
# it's a column vector, which is the shape our loss function will expect.
y = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1)

# --- Step 3: Define the Neural Network Model ---
# This is where we design our "brain". A class is a great way to organize it.
# It inherits from nn.Module, which is the base class for all PyTorch models.

class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        # The __init__ method is where you define the layers of your network.
        super(BinaryClassificationModel, self).__init__()

        # We'll create a simple network with one hidden layer.
        # A "layer" is like a stage of computation.

        # Layer 1: Takes the input features and transforms them.
        # nn.Linear is a standard "fully connected" layer.
        # It takes 'input_size' (2 in our case) features and outputs 'hidden_size' features.
        self.layer_1 = nn.Linear(input_size, hidden_size)

        # Activation Function: This decides which neurons "fire".
        # ReLU is a very common choice. It just turns any negative number into 0.
        self.relu = nn.ReLU()

        # Layer 2: This is our output layer.
        # It takes the 'hidden_size' features from the first layer and outputs a single value (1).
        self.layer_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # The "forward" method defines how data flows through the network.
        # It's the actual calculation step.

        # 1. Pass data through the first layer
        out = self.layer_1(x)
        # 2. Apply the activation function
        out = self.relu(out)
        # 3. Pass the result through the output layer
        out = self.layer_2(out)
        return out

# --- Step 4: Create the Model, Loss Function, and Optimizer ---

# Create an instance of our model.
# Input size is 2 (we have 2 features).
# Hidden size is 10 (this is a choice - 10 neurons in the hidden layer is a good start).
model = BinaryClassificationModel(input_size=num_features, hidden_size=10)

# Loss Function: This measures how wrong the model's predictions are.
# For binary classification, BCEWithLogitsLoss is the best choice. It's
# mathematically stable and designed for this exact problem.
criterion = nn.BCEWithLogitsLoss()

# Optimizer: This is the algorithm that updates the model's internal parameters
# (its "weights") to reduce the loss. It's how the model learns.
# 'Adam' is a very popular and effective optimizer. 'lr' is the learning rate.
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Step 5: The Training Loop ---
# This is where the magic happens. We'll show the model the data over and over,
# and it will slowly get better at predicting the correct labels.

num_epochs = 200 # An epoch is one full pass through the entire dataset.

for epoch in range(num_epochs):
    # 1. Forward pass: Get predictions from the model
    outputs = model(X)

    # 2. Calculate loss: Compare predictions with actual labels (y)
    loss = criterion(outputs, y)

    # 3. Backward pass and optimization
    optimizer.zero_grad() # Reset gradients from the previous step
    loss.backward()       # Calculate the gradients (how to change the weights)
    optimizer.step()      # Update the model's weights

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# --- Step 6: Evaluate the Model ---
# Now that the model is trained, let's see how well it does!

# We use `torch.no_grad()` because we're just evaluating, not training.
# This tells PyTorch not to calculate gradients, which makes it faster.
with torch.no_grad():
    # Get the model's raw output (called "logits")
    predictions_raw = model(X)

    # The output is a raw number. To turn it into a probability (0 to 1),
    # we use the sigmoid function.
    predictions_prob = torch.sigmoid(predictions_raw)

    # To get a final 0 or 1 prediction, we round the probabilities.
    # If prob > 0.5, it becomes 1. Otherwise, it becomes 0.
    predictions_class = predictions_prob.round()

    # Calculate accuracy: (number of correct predictions) / (total predictions)
    accuracy = (predictions_class.eq(y).sum() / float(y.shape[0]))
    print(f'\nTraining complete!')
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')

    # Let's test with a new, unseen data point
    new_point = torch.tensor([0.2, 0.3], dtype=torch.float32) # sum is 0.5, so true class is 0
    prediction = model(new_point)
    predicted_class = torch.sigmoid(prediction).round().item()
    print(f'\nTest on new point [0.2, 0.3]:')
    print(f'Predicted class: {int(predicted_class)}')


# --- Step 7: Visualize the Results ---
# A graph is a great way to see what the model has learned.
print('\nGenerating plot...')
with torch.no_grad():
    # Create a grid of points to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Get model predictions for every point on the grid
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    Z = model(grid_tensor)
    Z = torch.sigmoid(Z).reshape(xx.shape)
    Z = Z.round() # Get 0 or 1 predictions

    # Plot the decision boundary and the data points
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X_numpy[:, 0], X_numpy[:, 1], c=y_numpy, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Feature Data with Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

