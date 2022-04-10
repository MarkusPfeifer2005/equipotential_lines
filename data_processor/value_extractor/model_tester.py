import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


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
device = torch.device("cpu")
# Set model
model = torch.load(r"D:\OneDrive - brg14.at\Desktop\data_processor\value_extractor\models\lcd_cnn_20.pt").to(device)
model.eval()
# Set data
test_dataset = ImageFolder(root="../data/test_data", transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True,
                         pin_memory=device == torch.device("cuda"))


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
            print(f"{scores = }")
            print(f"{predictions = }")
            print(f"{y = }")
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")


if __name__ == "__main__":
    print("checking accuracy")
    check_accuracy(loader=test_loader, model=model)
