import os
import cv2
from build.data_accessories import RpiSession, MyImage, JSON, Directory, get_desktop
from PIL import Image
import matplotlib.pyplot as plt

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


class Sandbox:
    """
    The training data is an ImageFolder. This makes it possible to train with the old iInN.jpg format as well as
    the new i,n.jpg format.
    """
    def __init__(self, model: nn.Module, lr: float = 0.001, epochs: int = 20, batch_size: int = 64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_channels = 3
        self.num_classes = 10
        self.learning_rate = lr
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.train_path = r"D:\OneDrive - brg14.at\Desktop\train_data"

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_loader = self.get_loader()

    def train(self) -> None:
        self.model.train()
        for epoch in range(self.num_epochs):
            print(f"Epoch: {epoch}")
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                # Get data to cuda if possible
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)

                loss = self.criterion(self.model(data), targets)  # forward propagation
                self.optimizer.zero_grad()  # zero previous gradients
                loss.backward()  # back-propagation
                self.optimizer.step()  # gradient descent or adam step

    def get_loader(self) -> DataLoader:
        """
        See: https://www.youtube.com/watch?v=4JFVhJyTZ44&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=13
        :return DataLoader:
        """
        my_transforms = transforms.ToTensor()

        dataset = ImageFolder(root=self.train_path, transform=my_transforms)
        class_weights = []
        for root, subdir, files in os.walk(self.train_path):
            if len(files) > 0:  # if statement to avoid ZeroDivisionError
                class_weights.append(1 / len(files))

        sample_weights = [0] * len(dataset)

        for idx, (data, label) in enumerate(dataset):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                            pin_memory=self.device == torch.device("cuda"))  # shuffle=True not possible, due to weight
        return loader

    def evaluate(self, test_loader: DataLoader = None) -> None:
        test_loader = test_loader if test_loader else self.train_loader
        if test_loader == self.train_loader:
            print("Evaluating on train_loader!")

        num_correct = 0
        num_samples = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = self.model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}")
        self.model.train()

    def save_model(self, path: str = None) -> None:
        path = path if path else "models"
        path = os.path.join(path, f"lcd_cnn_{self.num_epochs}.pt")
        torch.save(self.model, path)
        print("successfully saved ")

    def __call__(self):
        """Just runs the methods in order when being called."""
        # todo: only save the model if it is better than before.
        self.train()
        self.evaluate()
        self.save_model()


class Session(RpiSession):
    """
    Special type of directory that contains a json and a csv.
    kwargs: path_to_dir
    If path_to_dir is not set, a filedialog box opens.
    """

    class MLPreparation:
        """
        This class can be called as a function. It takes an image that contains all 3 digits, splits it
        and yields 3 separate tensors.
        """

        def __init__(self):
            self.w = 150  # width
            self.h = 300  # height
            self.xy = [
                (215, 230),
                (310, 230),
                (400, 230)
            ]
            self.transformer = transforms.ToTensor()

        def __call__(self, img: MyImage) -> torch.tensor:
            img = img.get_matrix()  # Make a np.ndarray out of the MyImage.

            for x, y in self.xy:  # loop through all digit rois
                digit_roi = img[y - self.h // 2:y + self.h // 2, x - self.w // 2:x + self.w // 2]
                digit_roi = Image.fromarray(digit_roi)  # convert cv2 to PIL
                digit_roi = self.transformer(digit_roi)
                digit_roi = torch.unsqueeze(digit_roi, dim=0)
                yield digit_roi

    def __init__(self, **kwargs):
        super(Session, self).__init__(**kwargs)
        self.ml_preparation = self.MLPreparation()

    def fill_csv(self, model: nn.Module) -> None:
        """Completes the empty csv with data provided by cnn."""

        model.eval()

        for image in self.get_images():  # for each image in folder
            digits = []
            for digit in self.ml_preparation(image):  # for digit in image
                digits.append(int(model(digit).max(1)[-1][0]))  # append the predicted digit

            # row = [x,y,z,v]
            row_in_csv = list(image.get_label())
            row_in_csv.append(float(f"{digits[0]}.{digits[1]}{digits[2]}"))
            self.csv.append(row_in_csv)

    def prepare_for_ml(self, **kwargs):
        """
        Creates folder with images that can be used to broaden the capabilities of cnn. The method only works if
        a completely filled .csv file is available. The images must be in x,y,z.jpg format and the rois get saved in
        i,n.jpg format.
        """
        img_idx_json = {"img_idx": kwargs["img_idx"]} if "img_idx" in kwargs else JSON("build/image_index.json")

        # create folders
        ml_dir = Directory(os.path.join(kwargs["target_dir"], "ml_" + self.name)) if "target_dir" in kwargs \
            else Directory(os.path.join(get_desktop(), "ml_" + self.name))

        for i in range(10):
            Directory(os.path.join(ml_dir.path, str(i)))

        # fill folders
        for x, y, z, v in self.csv:  # Attention: v is a float value like: v.vv
            image = cv2.imread(os.path.join(self.path, f"{x},{y},{z}{self.image_ext}"))

            h, w = self.ml_preparation.h, self.ml_preparation.w
            xy = self.ml_preparation.xy

            for idx, (x1, y1) in enumerate(xy):
                digit1 = image[y1 - h // 2:y1 + h // 2, x1 - w // 2:x1 + w // 2]
                digit = str(v).replace('.', '')[idx]
                img_path = os.path.join(ml_dir.path, digit, f"i{img_idx_json['img_idx']}n{digit}.jpg")
                cv2.imwrite(img_path, digit1)
                img_idx_json["img_idx"] += 1


class Plot:
    def __init__(self, session: Session):
        self.session = session

    def plot(self) -> None:
        raise Exception("plot-function not implemented!")

    def __call__(self, *args, **kwargs):
        self.plot()


class TwoD(Plot):
    def __init__(self, session=None):
        super(TwoD, self).__init__(session)

    def plot(self) -> None:
        dummy_data = []
        for x in range(200):
            row = []
            for y in range(100):
                row.append(y)
            dummy_data.append(row)
            # todo: Everything under development!

        
        plt.show()


def main() -> None:
    model = torch.load(r"models/lcd_cnn_20.pt").to("cpu")

    active_session = Session()
    active_session.fill_csv(model=model)


if __name__ == "__main__":
    main()
