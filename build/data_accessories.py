#!/usr/bin/env python

"""
All classes for data handling and processing are located here.
In addition, this file can be called to create plots.
"""

import platform
import json
import csv
from tkinter import filedialog, Tk
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

from build.computervision import MyCNN


def get_desktop() -> str:  # todo: Generalize this function so it can be used on all systems!
    """Returns one of these 2 desktop paths dependent on the OS."""

    if platform.system() == "Windows":
        return r"D:\OneDrive - brg14.at\Desktop"
    elif platform.system() == "Linux":
        return r"/home/pi/Desktop"
    else:
        raise OSError("Can't determine the OS I'm running on!")


class File:
    """Only existing files can be opened."""

    def __init__(self, path_to_file: str):
        if os.path.isfile(path_to_file):
            self.path: str = path_to_file
        else:
            raise FileNotFoundError(f"Path '{path_to_file}' invalid!")

    def rename(self, new_path: str) -> None:
        """Useful for renaming and moving to another directory."""
        os.rename(self.path, new_path)
        self.path = new_path

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

    @property
    def label(self) -> tuple:
        """Returns a tuple with (x,y,z)."""
        label = self.name.replace(self.file_extension, '')
        return tuple(map(int, label.split(',')))

    @label.setter
    def label(self, new_label: tuple) -> None:
        """Renames image to include new data."""
        new_label = str(new_label)
        new_label = new_label.replace('(', '').replace(')', '').replace(' ', '')
        new_label += self.file_extension
        path = os.path.join(os.path.split(self.path)[0], new_label)
        self.rename(path)

    @property
    def matrix(self) -> np.ndarray:
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

    delimiter = ','

    def __init__(self, path_to_file: str):
        super(CSV, self).__init__(path_to_file)

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

    def append(self, entry: list) -> None:
        """
        Appends the item to the end of the file and saves the change. The entry must be a list and
        gets spread over a row.
        """
        with open(self.path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=self.delimiter)
            csv_writer.writerow(entry)

    def get_value(self, pos: list or tuple) -> float:
        """Returns the corresponding value to the given position."""
        with open(self.path, mode='r') as file:
            for i in csv.reader(file):
                if int(i[0]) == pos[0] and int(i[1]) == pos[1] and int(i[2]) == pos[2]:
                    return float(i[3])
        raise ValueError(f"Position '{pos}' does not exist!")


class Session:
    """
    Special type of directory that contains a json and a csv.
    kwargs: path_to_dir
    If path_to_dir is not set, a filedialog box opens.
    """

    def __init__(self, **kwargs):
        self.image_ext = kwargs["image_ext"] if "image_ext" in kwargs else ".jpg"
        if "path_to_dir" in kwargs:
            self.path = kwargs["path_to_dir"]
        else:
            Tk().withdraw()
            path_to_dir = filedialog.askdirectory()
            if path_to_dir != '' and path_to_dir != ():
                self.path = path_to_dir
            else:
                self.path = self.create_empty()

        self.csv = CSV(os.path.join(self.path, self.name + ".csv"))
        self.json = JSON(os.path.join(self.path, self.name + ".json"))

    def add_image(self, img: np.ndarray, pos: tuple) -> MyImage:
        """Adds the image to the session folder."""
        img_name = str(pos)
        img_name = img_name.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
        img_name += self.image_ext

        img_path = os.path.join(self.path, img_name)
        cv2.imwrite(filename=img_path, img=img)

        return MyImage(img_path)

    def read_images(self, decimal_pos: int, model: torch.nn.Module = None):
        """
        Reads the images and files the csv file. It does not care if data is already present,
        new data gets just appended.
        """
        if not model:
            model = torch.load(self.json["model"], map_location=torch.device("cpu"))
        model.eval()

        for image in tqdm(self.images):
            pos = image.label
            self.csv.append([pos[0], pos[1], pos[2], model.read(image.matrix, decimal_pos=decimal_pos)])

    def prepare_for_ml(self, **kwargs):
        """
        Creates folder with images that can be used to broaden the capabilities of cnn. The method only works if
        a completely filled .csv file is available. The images must be in x,y,z.jpg format and the rois get saved in
        i,n.jpg format.
        """
        w, h = 150, 300
        xy = [(215, 230), (310, 230), (400, 230)]
        img_idx_json = {"img_idx": kwargs["img_idx"]} if "img_idx" in kwargs else JSON("../build/image_index.json")

        # create folders
        ml_dir = os.path.join(kwargs["target_dir"], "ml_" + self.name) if "target_dir" in kwargs \
            else os.path.join(get_desktop(), "ml_" + self.name)
        os.mkdir(ml_dir)

        for i in range(10):
            os.mkdir(os.path.join(ml_dir, str(i)))

        # fill folders
        for x, y, z, v in self.csv:  # Attention: v is a float value like: v.vv
            image = cv2.imread(os.path.join(self.path, f"{x},{y},{z}{self.image_ext}"))

            for idx, (x1, y1) in enumerate(xy):
                digit_roi = image[y1 - h // 2:y1 + h // 2, x1 - w // 2:x1 + w // 2]  # digit roi
                digit = format(float(v), ".2f").replace('.', '')[idx]  # format makes it a string with 2 decimals
                img_path = os.path.join(ml_dir, digit, f"i{img_idx_json['img_idx']}n{digit}.jpg")
                cv2.imwrite(img_path, digit_roi)
                img_idx_json["img_idx"] += 1

    @property
    def images(self):
        """This generator yields all image files in the session as MyImage."""
        for file_name in os.listdir(path=self.path):
            if self.image_ext in file_name:
                yield MyImage(os.path.join(self.path, file_name))

    @classmethod
    def create_empty(cls, **kwargs) -> str:
        """
        Creates session folder with empty json and csv and returns the path of the session folder.
        If "path_to_dir" is provided in the kwargs this path gets used.
        This method is callable from the Image object without declaration.

        :return path:
        """
        master_path = kwargs["path_to_dir"] if "path_to_dir" in kwargs else get_desktop()
        session_name = cls.get_new_session_name(master_path)
        path = os.path.join(master_path, session_name)

        os.mkdir(path)
        open(os.path.join(path, session_name + ".json"), 'x').close()
        open(os.path.join(path, session_name + ".csv"), 'x').close()

        return path

    @staticmethod
    def get_new_session_name(path: str, folder_convention: str = "session") -> str:
        """Creates unique directory name like: 'session4'."""
        sessions = [folder for folder in os.listdir(path) if folder_convention in folder]
        if not sessions:
            return folder_convention + "0"
        else:
            largest_num = max((int(session.replace(folder_convention, '')) for session in sessions))
            return folder_convention + str(largest_num + 1)

    @property
    def name(self) -> str:
        return os.path.split(self.path)[-1]


class HeatMap:
    """A type of plot to display the measured electric potentials."""
    def __init__(self, session: Session):
        self.session = session

    def prepare_data(self) -> np.array:
        x_max, y_max, z_max = self.session.json["area_to_map"]
        x_stp, y_stp, z_stp = self.session.json["step_size"]

        data = [[[self.session.csv.get_value(pos=(x, y, z)) for x in range(0, x_max, x_stp)]
                 for y in range(y_max - y_stp, 0, -y_stp)] for z in tqdm(range(0, z_max, z_stp))]
        #            ^y is iterated in reverse

        return np.array(data)

    def plot(self) -> None:
        data = self.prepare_data()
        fig, ax = plt.subplots(nrows=1, ncols=len(data))
        planes = [i for i in range(0, self.session.json["area_to_map"][2], self.session.json["step_size"][2])]
        plt.tight_layout(pad=0.1)

        for i, (d, debt) in enumerate(zip(data, planes)):
            im = ax[i].imshow(d)
            ax[i].set_title(f"depth {debt}mm")
            ax[i].set_xlabel("x-distance [mm]")
            ax[i].set_ylabel("y-distance [mm]")

        try:
            plt.suptitle(f"voltage: {self.session.json['voltage']}\n"
                         f"electrode type: {self.session.json['electrode_type']}\n"
                         f"liquid: {self.session.json['liquid']}\n"
                         f"liquid temperature: {self.session.json['liquid_temp']}°C")
        except KeyError:
            pass

        plt.colorbar(im, ax=ax.ravel().tolist()).set_label("Voltage [V]")
        plt.show()


def main() -> None:
    session = Session()

    if len(session.csv) == 0:
        try:
            session.read_images(decimal_pos=1)  # If no images are available the csv remains untouched.
        except KeyError:  # If no model has been specified.
            pass

    p = HeatMap(session=session)
    p.plot()


if __name__ == "__main__":
    main()
