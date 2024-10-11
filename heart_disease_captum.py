import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import DeepLift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the heart disease dataset
data = pd.read_csv('datasets/heart.csv')  # Update with the correct path
X = data.drop(columns=["target"])  # Replace with the actual target column name
y = data["target"]  # Replace with the actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train.values)  # Use LongTensor for class labels
y_test_tensor = torch.LongTensor(y_test.values)  # Use LongTensor for class labels

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # Change to 3 for three classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize and train the model
input_dim = X_train.shape[1]
model = NeuralNetwork(input_dim)

criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

# Save the trained model state dictionary
torch.save(model.state_dict(), 'model_state_dict_heart_disease.pt')

# Load the trained model
model.eval()

# Initialize DeepLift
dl = DeepLift(model)

# Create a figure for plotting attributions
plt.figure(figsize=(10, 5))

# Loop through all target classes
for target_class in range(3):  # Looping through classes 0, 1, and 2
    attributions = []  # List to store attributions for each class

    for inputs, labels in test_loader:
        # Create baseline for each batch
        baseline = torch.zeros_like(inputs)
        attr = dl.attribute(inputs, baselines=baseline, target=target_class)
        attributions.append(attr.cpu().detach().numpy())

    # Convert list to numpy array and average attributions across all inputs
    attributions = np.concatenate(attributions, axis=0)
    average_attributions = np.mean(attributions, axis=0)

    # Normalize the attributions
    normalized_attributions = average_attributions / np.sum(np.abs(average_attributions))

    # Plot feature importance for the current target class
    feature_names = X.columns.tolist()  # Use the actual feature names from the dataset
    sorted_indices = np.argsort(normalized_attributions)[::-1]
    sorted_attributions = normalized_attributions[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]

    plt.barh(range(len(sorted_attributions)), sorted_attributions, align='center', alpha=0.6, label=f'Class {target_class}')
    plt.yticks(range(len(sorted_attributions)), sorted_feature_names)
    plt.xlabel('Normalized Average Attribution')
    plt.title('Feature Importance Using DeepLift for Target Classes')
    plt.legend()

plt.show()