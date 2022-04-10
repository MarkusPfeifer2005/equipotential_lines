import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os


class MyCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.fc1 = nn.Linear(44400, num_classes)  # used: print(f"{x.shape = }") to determine 307200

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
num_epochs = 20
batch_size = 64


# Load Data
def get_loader(root_dir: str, batch_size: int):
    my_transforms = transforms.ToTensor()

    dataset = ImageFolder(root=root_dir, transform=my_transforms)
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:  # if statement to avoid ZeroDivisionError
            class_weights.append(1/len(files))

    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        pin_memory=device == torch.device("cuda"))  # shuffle=True not possible, due to weight
    return loader


train_loader = get_loader(root_dir="../data/train_data", batch_size=batch_size)
test_dataset = ImageFolder(root="../data/test_data", transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                         pin_memory=device == torch.device("cuda"))

# Initialize the network
model = MyCNN(in_channels=in_channels, num_classes=num_classes).to(device)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the network
model.train()
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward propagation
        scores = model(data)
        loss = criterion(scores, targets)

        # zero previous gradients
        optimizer.zero_grad()

        # back-propagation
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")
    model.train()


print("Checking on training data.")
check_accuracy(train_loader, model)
print("Checking on test data.")
check_accuracy(test_loader, model)

torch.save(model, f"models/lcd_cnn_{num_epochs}.pt")  # saving the trained model
