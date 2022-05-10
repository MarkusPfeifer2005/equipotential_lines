import os
import shutil
import json
import csv
import cv2
import numpy as np
from tkinter import filedialog, Tk
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class MyCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
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


class File:
    """Only existing files can be opened."""

    def __init__(self, path_to_file: str):
        if os.path.isfile(path_to_file):
            self.path: str = path_to_file
        else:
            raise FileNotFoundError(f"Path '{path_to_file}' invalid!")

    def rename(self, new_path: str) -> None:
        """Useful for renaming and moving to another location."""
        os.rename(self.path, new_path)
        self.path = new_path

    def delete(self):
        os.remove(self.path)
        print(f"Deleted file from {self.path}")

    @property
    def name(self) -> str:
        """Returns the filename with the file extension."""
        return os.path.split(self.path)[-1]


class MyImage(File):
    """
    File that can be displayed as matrix. It has a label of x,y,z format. Only existing .jpg files can be opened.
    """
    file_extension: str = ".jpg"

    def __init__(self, path_to_file: str):
        if self.file_extension not in path_to_file:
            raise ValueError(f"File is no {self.file_extension}!")
        super(MyImage, self).__init__(path_to_file)

    def get_label(self) -> tuple:
        """Returns a tuple with (x,y,z)."""
        label = self.name.replace(self.file_extension, '')
        return tuple(map(int, label.split(',')))

    def set_label(self, new_label: tuple) -> None:
        """Renames image to include new data."""
        new_label = str(new_label)
        new_label = new_label.replace('(', '').replace(')', '').replace(' ', '')
        new_label += self.file_extension
        path = os.path.join(os.path.split(self.path)[0], new_label)
        self.rename(path)

    def get_matrix(self) -> np.ndarray:
        return cv2.imread(self.path)


class JSON(File):
    """A savable dictionary."""

    def __init__(self, path_to_file: str):
        super(JSON, self).__init__(path_to_file)

    def __getitem__(self, item):
        with open(self.path, 'r') as file:
            return json.load(file)[item]

    def __setitem__(self, key, value):
        with open(self.path, "r") as file:
            try:
                data = json.load(file)
            except json.decoder.JSONDecodeError:  # file is empty
                data = {}
            finally:
                data[key] = value

        with open(self.path, "w") as file:
            json.dump(data, file)

    def __delitem__(self, key):
        with open(self.path, "r") as file:
            data: dict = json.load(file)
        del data[key]
        with open(self.path, "w") as file:
            json.dump(data, file)

    def __len__(self):
        try:
            with open(self.path, 'r') as file:
                return len(json.load(file))
        except json.decoder.JSONDecodeError:
            return 0


class CSV(File):
    """
    A savable list.
    If it is used for the session data it would look like this: |x:int|y:int|z:int|v:float|.
    """
    # todo: replace the open('r') open('w') with a single open() like open('r+) or open('w+)

    delimiter = ','

    def __init__(self, path_to_file: str):
        super(CSV, self).__init__(path_to_file)

    def append(self, entry: list) -> None:
        """
        Appends the item to the end of the file and saves the change. The entry must be a list and
        gets spread over a row.
        """
        with open(self.path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=self.delimiter)
            csv_writer.writerow(entry)

    def __iter__(self):
        """Extract values from csv file and yields them in the form of a list."""
        with open(self.path, mode='r') as file:
            for i in csv.reader(file):
                yield i

    def __getitem__(self, idx):
        with open(self.path, mode='r') as file:
            return list(csv.reader(file))[idx]

    def __len__(self):
        with open(self.path, mode='r') as file:
            return len(list(csv.reader(file)))

    def __setitem__(self, idx, value):
        with open(self.path, mode='r') as file:
            data = list(csv.reader(file))
        data[idx] = value
        with open(self.path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=self.delimiter)
            writer.writerows(data)

    def __delitem__(self, idx):
        with open(self.path, mode='r') as file:
            data = list(csv.reader(file))
        del data[idx]
        with open(self.path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=self.delimiter)
            writer.writerows(data)


class Directory:
    def __init__(self, path_to_dir: str):
        self.path = path_to_dir
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
            print(f"Created directory at '{self.path}'")

    def delete(self):
        """Delete directory and content. The instance remains"""
        shutil.rmtree(self.path)
        print(f"Deleted directory (and content) from {self.path}")

    def get_files(self) -> list:
        """Returns a list containing all the names of the files contained in the directory."""
        return os.listdir(self.path)

    @property
    def name(self) -> str:
        return os.path.split(self.path)[-1]


class Sandbox:
    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_channels = 3
        self.num_classes = 10
        self.learning_rate = 0.001
        self.num_epochs = 20
        self.batch_size = 64
        self.train_path = r"D:\OneDrive - brg14.at\Desktop\train_data"

        self.model = model
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

    def save_model(self) -> None:
        torch.save(self.model, f"models/lcd_cnn_{self.num_epochs}.pt")
        print("successfully saved ")

    def __call__(self):
        """Just runs the methods in order when being called."""
        self.train()
        self.evaluate()
        self.save_model()


class Session(Directory):
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
        self.image_ext = kwargs["image_ext"] if "image_ext" in kwargs else ".jpg"
        if "path_to_dir" in kwargs:
            super(Session, self).__init__(kwargs["path_to_dir"])
        else:
            Tk().withdraw()
            path_to_dir = filedialog.askdirectory()
            if path_to_dir != '':
                super(Session, self).__init__(path_to_dir)
            else:
                super(Session, self).__init__(self.create_empty())

        self.csv = CSV(os.path.join(self.path, self.name + ".csv"))
        self.json = JSON(os.path.join(self.path, self.name + ".json"))

        self.ml_preparation = self.MLPreparation()

    @classmethod
    def create_empty(cls, **kwargs) -> str:
        """
        Creates session folder with empty json and csv and returns the path of the session folder.
        If "path_to_dir" is provided in the kwargs this path gets used.
        This method is callable from the Image object without declaration.

        :return path:
        """
        master_path = kwargs["path_to_dir"] if "path_to_dir" in kwargs else cls.get_desktop()
        session_name = cls.get_new_session_name(master_path)
        path = os.path.join(master_path, session_name)

        os.mkdir(path)
        open(os.path.join(path, session_name + ".json"), 'x').close()
        open(os.path.join(path, session_name + ".csv"), 'x').close()

        return path

    @staticmethod
    def get_new_session_name(path: str, folder_convention: str = "session") -> str:
        """Creates unique directory name like: 'session4'."""
        largest_num = 0
        for directory in os.listdir(path):
            if folder_convention in directory:
                directory = directory.replace(folder_convention, '')
                if int(directory) > largest_num:
                    largest_num = int(directory)
        return folder_convention + str(largest_num + 1)

    @staticmethod
    def get_desktop() -> str:
        """
        ONLY TESTED ON MY WINDOWS!!
        must be changed for raspberry

        :return r"D:\OneDrive - brg14.at\Desktop":
        """
        # return os.path.join(os.environ["HOMEPATH"], "Desktop")
        # return str(os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop'))
        return r"D:\OneDrive - brg14.at\Desktop"

    def get_images(self):
        """
        This generator yields all image files in the session.
        The files get passed on in the custom MyImage format.
        """
        for file_name in os.listdir(path=self.path):
            if self.image_ext in file_name:
                yield MyImage(os.path.join(self.path, file_name))

    def add_image(self, img: np.ndarray, pos: tuple) -> MyImage:
        """Adds the image to the session folder and updates the position value in the json file."""
        img_name = str(pos).replace('(', '').replace(')', '').replace(' ', '') + self.image_ext
        img_path = os.path.join(self.path, img_name)
        cv2.imwrite(filename=img_path, img=img)

        self.json["last_pos"] = pos
        return MyImage(img_path)

    def del_image(self, img: MyImage = None, img_name: str = None) -> None:
        """
        Image gets deleted from session. If the image is not part of the session an Exception is raised! The image can
        be specified via the Image object or the path to the image.
        """

        if img:
            if os.path.split(img.path)[0] == self.path:  # Can't be chained with "and" because img could be None!
                img.delete()
        elif img_name:
            os.remove(os.path.join(self.path, img_name))
        else:
            raise ValueError("No data about image provided!")

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
            else Directory(os.path.join(self.get_desktop(), "ml_" + self.name))

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


def main() -> None:
    active_session = Session()  # new empty session is created!

    # step 2 - complete the csv file with the help of the neural net
    model = torch.load(r"models/lcd_cnn_20.pt")
    model.to(device="cpu")

    active_session.fill_csv(model=model)

    # step 3 - provide new data for neural net
    active_session.prepare_for_ml()

    # step 4 (optional) - train neural net with the new data
    pass


if __name__ == "__main__":
    main()
