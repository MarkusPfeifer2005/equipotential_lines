#!/usr/bin/env python

"""Everything having to do with machinelearning is handled via this file."""

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from torchvision.datasets import ImageFolder


class MyCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(MyCNN, self).__init__()

        # neuron-layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(44400, num_classes)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = to_tensor(x)
            x = torch.unsqueeze(x, dim=0)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def save(self, name: str, path: str = None) -> None:
        """Saves model to file. Extension '.pt'. The prefix 'lcd_cnn_' gets added to the name."""
        name = name.split('.')[0]
        path = os.path.join(path if path else "../models", f"lcd_cnn_{name}.pt")
        torch.save(self, path)
        print(f"Successfully saved model to {path}.")

    def classify(self, img: np.ndarray) -> int:
        """Classifies single digit."""
        predictions = self(img)
        probability, digit = predictions.max(1)
        probability, digit = probability.item(), digit.item()
        return digit

    def read(self, img: np.ndarray) -> float:
        """Reads entire screen of multimeter."""
        w, h = 150, 300
        xy = [(215, 230), (310, 230), (400, 230)]

        predictions = []
        for x, y in xy:  # loop through all digit rois
            digit_roi = img[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
            self.classify(digit_roi)
            predictions.append(self.classify(digit_roi))

        return float(f"{predictions[0]}.{predictions[1]}{predictions[2]}")


class MyLoader(DataLoader):
    """See: https://www.youtube.com/watch?v=4JFVhJyTZ44&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=13"""

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomAutocontrast()
    ])

    def __init__(self, train_path: str, batch_size: int, device: torch.device):

        print("creating dataset...")
        dataset = ImageFolder(root=train_path, transform=self.transforms)

        # OLD:
        # class_weights = []
        # for root, subdir, files in os.walk(train_path):
        #     if len(files) > 0:  # if statement to avoid ZeroDivisionError
        #         class_weights.append(1 / len(files))
        # NEW:
        print("getting class weights...")
        walker = os.walk(train_path)
        next(walker)  # skips item 0 which is the root-directory itself
        class_weights = [len(files) for dir_path, classes, files in walker]

        # specify weight for each individual image
        # OLD:
        # sample_weights = [0] * len(dataset)
        # for idx, (data, label) in enumerate(dataset):
        #     sample_weights[idx] = class_weights[label]
        # NEW:
        print("creating individual sample weights...")
        sample_weights = [class_weights[label] for idx, (data, label) in enumerate(dataset)]

        print("setting up ")
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

    with torch.no_grad():

        num_correct = 0
        num_samples = 0

        for ipt, targets in tqdm(test_loader):
            ipt, targets = ipt.to(device=device), targets.to(device=device)

            predictions = model(ipt)
            predictions = predictions.max(1)  # returns tensor(maximums, indices of maximums)
            _, indices = predictions  # indices are correspondent to the numbers (_ are the maximums)
            correct = torch.eq(indices, targets)  # element wise comparison

            num_correct += torch.sum(correct)  # adding correct ones
            num_samples += indices.size(0)

    success_ratio = float(num_correct) / float(num_samples) * 100
    print(f"Got {num_correct} / {num_samples} with accuracy {success_ratio:.2f}%.")
    return success_ratio


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = MyLoader(device=device, train_path=r"D:\OneDrive - brg14.at\Desktop\train_data", batch_size=64)
    # test_loader = MyLoader(device=device, train_path=r"D:\OneDrive - brg14.at\Desktop\test_data", batch_size=64)
    test_loader = train_loader

    for epoch in [5]:
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
