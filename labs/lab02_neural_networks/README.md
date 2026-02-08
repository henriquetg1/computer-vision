# Lab 2: Neural Networks with PyTorch

**Due Date:** End of Week 2  
**Points:** 100  
**Estimated Time:** 1.5 hours in class + 2 hours homework

## Learning Objectives

- Understand PyTorch tensors and automatic differentiation
- Implement a simple neural network from scratch
- Train a network using gradient descent
- Use PyTorch's `nn.Module` for cleaner code
- Visualize training progress and decision boundaries

## Prerequisites

- Completed Lab 1
- Basic understanding of calculus (derivatives)
- Linear algebra basics (matrix multiplication)

## Setup

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

## Part 1: PyTorch Fundamentals (30 min)

### Task 1.1: Tensor Operations

**File:** `task1_tensors.ipynb`

Explore PyTorch tensor operations:

```python
import torch
import numpy as np

# Create tensors different ways
t1 = torch.tensor([1, 2, 3])
t2 = torch.zeros((3, 3))
t3 = torch.randn((2, 4))  # Random normal distribution

# Your tasks:
# 1. Create a 5x5 tensor of ones
# 2. Create a tensor from a NumPy array
# 3. Perform element-wise operations
# 4. Matrix multiplication
# 5. Reshape operations
```

**Key operations to understand:**
- `torch.zeros`, `torch.ones`, `torch.randn`
- `tensor.shape`, `tensor.dtype`
- `tensor.view()` and `tensor.reshape()`
- `torch.mm()` vs `torch.matmul()` vs `@`
- Moving tensors to GPU: `tensor.to('cuda')`

### Task 1.2: Automatic Differentiation

Understand PyTorch's autograd:

```python
# Create tensor with gradient tracking
x = torch.tensor([2.0], requires_grad=True)

# Define a function
y = x ** 2 + 3 * x + 1

# Compute gradient
y.backward()

print(f"dy/dx at x=2: {x.grad}")  # Should be 2*2 + 3 = 7
```

**Your tasks:**
1. Compute gradient of `f(x) = x^3 - 2x^2 + x` at x=1
2. Compute partial derivatives of `f(x,y) = x^2 + xy + y^2`
3. Verify your results match hand calculations

### Task 1.3: Gradient Descent from Scratch

Implement gradient descent to minimize `f(x) = (x - 3)^2`:

```python
# Initialize
x = torch.tensor([0.0], requires_grad=True)
learning_rate = 0.1
num_iterations = 100

# Store history
history = []

for i in range(num_iterations):
    # Forward pass
    y = (x - 3) ** 2
    
    # Backward pass
    y.backward()
    
    # Update (no_grad to prevent tracking)
    with torch.no_grad():
        x -= learning_rate * x.grad
        history.append(x.item())
    
    # Clear gradients
    x.grad.zero_()
    
# Plot convergence
import matplotlib.pyplot as plt
plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Gradient Descent Convergence')
plt.show()
```

## Part 2: Build a Neural Network from Scratch (40 min)

### Task 2.1: Manual Neural Network

Implement a 2-layer network manually:

**File:** `task2_manual_nn.ipynb`

```python
import torch
import torch.nn.functional as F

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize network weights
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden neurons
            output_size: Number of output classes
        """
        # Initialize weights with small random values
        self.W1 = torch.randn(input_size, hidden_size) * 0.01
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(hidden_size, output_size) * 0.01
        self.b2 = torch.zeros(output_size)
        
        # Make them require gradients
        self.W1.requires_grad = True
        self.b1.requires_grad = True
        self.W2.requires_grad = True
        self.b2.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, input_size)
        
        Returns:
            Output tensor (batch_size, output_size)
        """
        # Layer 1: Linear + ReLU
        h = torch.matmul(x, self.W1) + self.b1
        h = F.relu(h)
        
        # Layer 2: Linear (logits)
        out = torch.matmul(h, self.W2) + self.b2
        
        return out
    
    def parameters(self):
        """Return list of all parameters"""
        return [self.W1, self.b1, self.W2, self.b2]
```

**Your task:** Extend this to include:
1. A sigmoid activation option
2. Multiple hidden layers
3. Batch normalization (optional)

### Task 2.2: Train on Synthetic Data

Create a simple classification dataset and train your network:

```python
from sklearn.datasets import make_moons, make_circles
import numpy as np

# Create dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X)
y_train = torch.LongTensor(y)

# Initialize network
net = SimpleNN(input_size=2, hidden_size=10, output_size=2)

# Training loop
learning_rate = 0.01
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    logits = net.forward(X_train)
    loss = F.cross_entropy(logits, y_train)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    with torch.no_grad():
        for param in net.parameters():
            param -= learning_rate * param.grad
            param.grad.zero_()
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

**Visualization task:** Plot the decision boundary at different epochs

### Task 2.3: PyTorch `nn.Module` Way

Reimplement using PyTorch's proper class structure:

```python
import torch.nn as nn

class BetterNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BetterNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize
model = BetterNN(2, 10, 2)

# Use PyTorch optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    # Forward
    outputs = model(X_train)
    loss = F.cross_entropy(outputs, y_train)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Compare:** Train both versions and compare:
- Training time
- Final accuracy
- Code simplicity

## Part 3: MNIST Digit Classification (50 min)

### Task 3.1: Load MNIST Dataset

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Download and load training data
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

**Exploration task:**
1. Display sample images from each class
2. Check dataset size
3. Verify the normalization

### Task 3.2: Build and Train MNIST Classifier

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten image
        x = x.view(-1, 28 * 28)
        
        # Hidden layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        return x

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                  f' Loss: {loss.item():.6f}')

# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return accuracy

# Main training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# Save model
torch.save(model.state_dict(), 'mnist_model.pth')
```

### Task 3.3: Analysis and Visualization

1. **Plot training curves:**
   - Loss over epochs
   - Accuracy over epochs

2. **Confusion matrix:**
   - Which digits are most confused?

3. **Error analysis:**
   - Display misclassified examples
   - What patterns do you notice?

4. **Experiment with:**
   - Different learning rates: [0.0001, 0.001, 0.01, 0.1]
   - Different architectures: more/fewer layers
   - Different activation functions: ReLU vs Tanh vs Sigmoid
   - Different optimizers: SGD vs Adam vs RMSprop

## Homework: Extended MNIST Experiments

### Part A: Architecture Search (40 points)

Design and compare at least 3 different network architectures:

1. **Shallow network:** 1-2 hidden layers
2. **Deep network:** 4-5 hidden layers
3. **Wide network:** Fewer layers but many neurons

Create a table comparing:
- Number of parameters
- Training time per epoch
- Final test accuracy
- Overfitting behavior

### Part B: Visualization (30 points)

1. **Weight visualization:**
   - Visualize the weights of the first layer as images
   - What patterns do you see?

2. **Activation visualization:**
   - Pass an image through the network
   - Visualize activations at each layer

3. **Decision boundary:**
   - Use t-SNE or PCA to reduce features to 2D
   - Visualize the learned feature space

### Part C: Implement Momentum SGD (30 points)

Manually implement SGD with momentum (don't use PyTorch's optimizer):

```python
class MomentumSGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        """
        Initialize momentum optimizer
        
        Args:
            parameters: Network parameters
            lr: Learning rate
            momentum: Momentum coefficient (0-1)
        """
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.parameters]
    
    def step(self):
        """
        Update parameters using momentum
        """
        # YOUR IMPLEMENTATION HERE
        pass
    
    def zero_grad(self):
        """Clear gradients"""
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()
```

Compare your implementation with PyTorch's `torch.optim.SGD(momentum=0.9)`.

## Submission

Submit a ZIP file containing:
1. All completed notebooks
2. Trained model file (`mnist_model.pth`)
3. PDF report with:
   - Architecture comparison table
   - All required visualizations
   - Answers to analysis questions
   - Code for momentum optimizer

**Naming:** `lab2_yourname.zip`

## Grading Rubric

| Component | Points |
|-----------|--------|
| Part 1: PyTorch Fundamentals | 15 |
| Part 2: Manual NN | 25 |
| Part 3: MNIST Classifier | 30 |
| Homework Part A: Architecture | 15 |
| Homework Part B: Visualization | 10 |
| Homework Part C: Momentum | 5 |
| **Total** | **100** |

## Common Issues and Solutions

**Issue:** Out of memory error
- **Solution:** Reduce batch size or model size

**Issue:** Loss is NaN
- **Solution:** Lower learning rate, check for division by zero

**Issue:** Model not improving
- **Solution:** Check data normalization, learning rate, initialization

**Issue:** Training is very slow
- **Solution:** Use GPU, reduce model size, use smaller dataset for testing

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](http://www.deeplearningbook.org/) - Chapter 6
- [CS231n notes on Neural Networks](http://cs231n.github.io/)

## Bonus Challenges

1. **Early stopping:** Implement early stopping based on validation loss
2. **Learning rate schedule:** Reduce learning rate when loss plateaus
3. **Data augmentation:** Add random rotations/shifts to MNIST
4. **Ensemble:** Train multiple models and combine predictions

Good luck! 🧠🔥
