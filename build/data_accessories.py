import os
import platform
import shutil
import json
import csv
import cv2
import numpy as np
from tkinter import filedialog, Tk


def get_desktop() -> str:
    """
    Returns one of these 2 desktop paths dependent on the OS.
    :return r"D:\OneDrive - brg14.at\Desktop" or "/home/pi/Desktop":
    """
    if platform.system() == "Windows":
        return r"D:\OneDrive - brg14.at\Desktop"
    elif platform.system() == "Linux":
        return r"/home/pi/Desktop"
    else:
        raise OSError("Can't determine the os I'm running on!")


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


class RpiSession(Directory):
    def __init__(self, **kwargs):
        self.image_ext = kwargs["image_ext"] if "image_ext" in kwargs else ".jpg"
        if "path_to_dir" in kwargs:
            super(RpiSession, self).__init__(kwargs["path_to_dir"])
        else:
            Tk().withdraw()
            path_to_dir = filedialog.askdirectory()
            if path_to_dir != '' and path_to_dir != ():
                super(RpiSession, self).__init__(path_to_dir)
            else:
                super(RpiSession, self).__init__(self.create_empty())

        self.csv = CSV(os.path.join(self.path, self.name + ".csv"))
        self.json = JSON(os.path.join(self.path, self.name + ".json"))

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
        largest_num = max((int(f.replace(folder_convention, '')) for f in os.listdir(path) if folder_convention in f))
        return folder_convention + str(largest_num + 1)

    def get_images(self):
        """This generator yields all image files in the session as MyImage."""
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
            if os.path.split(img.path)[0] == self.path:  # Can not be chained with "and" because img could be None!
                img.delete()
        elif img_name:
            os.remove(os.path.join(self.path, img_name))
        else:
            raise ValueError("No data about image provided!")
