"""Just copy file into folder and run it!"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import cv2
from PIL import Image
import os
import csv


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


def prepare_for_my_net(img) -> torch.tensor:
    # make tensor
    transformer = transforms.ToTensor()
    img = transformer(img)
    img = torch.unsqueeze(img, dim=0)

    return img


if __name__ == "__main__":
    # load model
    device = torch.device("cpu")
    model = torch.load(r"D:\OneDrive - brg14.at\Desktop\data_processor\value_extractor\models\lcd_cnn.pt").to(device)
    model.eval()

    # roi values
    w = 150
    h = 300
    xy = [
        (215, 230),
        (310, 230),
        (400, 230)
    ]

    # get relevant images (.jpg)
    jpg_files = [file for file in os.listdir() if ".jpg" in file and 'x' in file and 'y' in file and 'v' not in file]

    # label the images
    labels = []
    for image_name in jpg_files:
        img = cv2.imread(image_name)

        # get values
        values = []
        for x, y in xy:
            digit_roi = img[y-h//2:y+h//2, x-w//2:x+w//2]
            digit_roi = Image.fromarray(digit_roi)  # convert cv2 to PIL
            digit_roi = prepare_for_my_net(digit_roi)
            out = int(model(digit_roi).max(1)[-1][0])
            values.append(out)

        x = image_name.split('x')[1].split('y')[0]
        y = image_name.split('y')[1].replace(".jpg", '')
        v = f"{values[0]}.{values[1]}{values[2]}"  # add decimal point and make string

        labels.append([x, y, v])

    # save labels to csv
    # get current session number
    session = ""
    for char in os.getcwd().split("\\")[-1]:
        if char.isdigit():
            session += char
    session = int(session)

    # write to file
    with open(f"session{session}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(labels)
