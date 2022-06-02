import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class MyCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(MyCNN, self).__init__()

        # neuron-layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.fc1 = nn.Linear(44400, num_classes)  # used: print(f"{x.shape = }") to determine 44400

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def save(self, name: str, path: str = None) -> None:
        """Saves model to file."""
        name = name.split('.')[0]  # prevent wrong naming convention
        path = os.path.join(path if path else "../models", f"lcd_cnn_{name}.pt")
        torch.save(self, path)
        print(f"Successfully saved model to {path}.")


class MyLoader(DataLoader):
    """
    See: https://www.youtube.com/watch?v=4JFVhJyTZ44&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=13
    :return DataLoader:
    """

    def __init__(self, train_path: str, batch_size: int, device: torch.device):
        my_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-5, 5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomAutocontrast()
        ])

        dataset = ImageFolder(root=train_path, transform=my_transforms)

        class_weights = []
        for root, subdir, files in os.walk(train_path):
            if len(files) > 0:  # if statement to avoid ZeroDivisionError
                class_weights.append(1 / len(files))

        sample_weights = [0] * len(dataset)
        for idx, (data, label) in enumerate(dataset):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, pin_memory=device == torch.device("cuda"))


def train(model: nn.Module, num_epochs: int, train_loader: DataLoader, criterion,
          optimizer: optim, device: torch.device) -> None:
    """Trains the model. The model is put into training-mode."""
    model.train()

    for _ in tqdm(range(num_epochs)):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device=device), targets.to(device=device)  # Get data to cuda if possible

            loss = criterion(model(data), targets)  # forward propagation
            optimizer.zero_grad()  # zero previous gradients
            loss.backward()  # back-propagation
            optimizer.step()  # gradient descent or adam step


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluates the accuracy of the model. The model is put into evaluation-mode.
    :return float (accuracy)
    """
    model.eval()

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device=device), y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}%.")
    return float(num_correct) / float(num_samples) * 100


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = MyLoader(device=device, train_path=r"D:\OneDrive - brg14.at\Desktop\train_data", batch_size=64)
    test_loader = MyLoader(device=device, train_path=r"D:\OneDrive - brg14.at\Desktop\test_data", batch_size=64)

    for epoch in [5, 10]:
        print(f"========= epochs: {epoch} =========")

        model = MyCNN().to(device)  # always creates a completely new model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train(model=model, device=device, num_epochs=epoch, train_loader=train_loader,
              criterion=criterion, optimizer=optimizer)
        evaluate(model=model, test_loader=train_loader, device=device)
        accuracy = evaluate(model=model, test_loader=test_loader, device=device)
        if accuracy > 99:
            model.save(name=f"{epoch}_{int(accuracy)}")


if __name__ == "__main__":
    main()
