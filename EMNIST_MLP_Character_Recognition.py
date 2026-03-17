# EMNIST_MLP_Character_Recognition.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------------
# 0. Setup
# ---------------------------
print("Python script started...")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# 1. Define Transform
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize pixel values
])

# ---------------------------
# 2. Load EMNIST Letters Dataset
# ---------------------------
train_dataset = datasets.EMNIST(
    root='./data',
    split='letters',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.EMNIST(
    root='./data',
    split='letters',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("EMNIST Letters dataset loaded successfully!")

# ---------------------------
# 3. Define MLP Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 26)  # 26 letters

    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten images
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP().to(device)

# ---------------------------
# 4. Loss and Optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 5. Training Loop
# ---------------------------
epochs = 5  # increase for better accuracy
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), (labels - 1).to(device)  # shift labels 1-26 -> 0-25
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# ---------------------------
# 6. Evaluate Model
# ---------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), (labels - 1).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100*correct/total:.2f}%")

# ---------------------------
# 7. Visualize a Sample Prediction
# ---------------------------
image, label = test_dataset[0]
model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
    _, predicted = torch.max(output, 1)

# EMNIST Letters labels: 1-26
actual_letter = chr(label + 64)      # 1->'A', 2->'B', etc.
predicted_letter = chr(predicted.item() + 65)  # 0->'A', 1->'B', etc.

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Actual: {actual_letter}, Predicted: {predicted_letter}")
plt.show()