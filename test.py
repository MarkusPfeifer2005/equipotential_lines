#!/usr/bin/env python

"""
Here you find all unit tests.
Not every function and class gets tested, since there are some tests that would require the machine to operate.
"""

import unittest
import os
import cv2
import shutil
import numpy as np

from equipotential_lines.data_accessories import get_session_folder, File, MyImage, JSON, CSV, Session
from equipotential_lines.computervision import MyCNN


class TestFile(unittest.TestCase):
    def setUp(self):
        """Runs before each test. Each of the instances gets freshly created for each test."""
        try:
            open("test_files/file1.txt", 'x').close()  # create file
        except FileExistsError:
            pass
        self.file1 = File("test_files/file1.txt")

    def tearDown(self):
        """Runs after each test."""
        try:
            os.remove("test_files/file1.txt")
        except FileNotFoundError:
            pass

        try:
            os.remove("test_files/renamed.txt")
        except FileNotFoundError:
            pass

    def test_path(self):
        self.assertEqual(self.file1.path, "test_files/file1.txt")

    def test_name(self):
        self.assertEqual(self.file1.name, "file1.txt")

    def test_rename(self):
        self.file1.rename("test_files/renamed.txt")


class TestMyImage(unittest.TestCase):
    def setUp(self):
        shutil.copyfile("test_files/session3/0,0,0.jpg", "test_files/0,0,0.jpg")
        self.img1 = MyImage("test_files/0,0,0.jpg")

    def tearDown(self):
        try:
            os.remove("test_files/0,0,0.jpg")
        except FileNotFoundError:
            pass

        try:
            os.remove("test_files/1,1,1.jpg")
        except FileNotFoundError:
            pass

    def test_path(self):
        self.assertEqual(self.img1.path, "test_files/0,0,0.jpg")

    def test_name(self):
        self.assertEqual(self.img1.name, "0,0,0.jpg")
        self.img1.label = (1, 1, 1)
        self.assertEqual(self.img1.label, (1, 1, 1))

    def test_get_label(self):
        self.assertEqual(self.img1.label, (0, 0, 0))

    def test_set_label(self):
        self.img1.label = (1, 1, 1)
        self.assertEqual(self.img1.label, (1, 1, 1))

    def test_get_matrix(self):
        self.assertIsInstance(self.img1.matrix, np.ndarray)


class TestJSON(unittest.TestCase):
    def setUp(self):
        shutil.copyfile("test_files/session3/session3.json", "test_files/session3.json")
        self.json1 = JSON("test_files/session3.json")

    def tearDown(self):
        try:
            os.remove("test_files/session3.json")
        except FileNotFoundError:
            pass

    def test___getitem__(self):
        self.assertEqual(self.json1["area_to_map"], [255, 150, 10])
        self.assertEqual(self.json1["step_size"], [5, 5, 10])
        self.assertEqual(self.json1["last_pos"], [0, 45, 0])
        self.assertEqual(self.json1["voltage"], "")
        self.assertEqual(self.json1["electrode_type"], "")
        self.assertEqual(self.json1["liquid"], "")

    def test___setitem__(self):
        self.json1["area_to_map"] = [0, 0, 0]
        self.assertEqual(self.json1["area_to_map"], [0, 0, 0])

    def test___len__(self):
        self.assertEqual(len(self.json1), 6)

    def test___delitem__(self):
        del self.json1["step_size"]
        self.assertEqual(len(self.json1), 5)


class TestCSV(unittest.TestCase):
    def setUp(self):
        shutil.copyfile("test_files/session3/session3.csv", "test_files/session3.csv")
        self.csv1 = CSV("test_files/session3.csv")

    def tearDown(self):
        try:
            os.remove("test_files/session3.csv")
        except FileNotFoundError:
            pass

    def test__getitem__(self):
        self.assertEqual(self.csv1[0], ['0', '0', '0', '2.07'])
        self.assertEqual(self.csv1[5], ['0', '25', '0', '2.06'])

    def test_append(self):
        self.csv1.append([1, 2, 3, 4])
        self.assertEqual(self.csv1[10], ['1', '2', '3', '4'])

    def test___iter__(self):
        targets = [
            ['0', '0', '0', '2.07'],
            ['0', '5', '0', '2.07'],
            ['0', '10', '0', '2.07'],
            ['0', '15', '0', '2.07'],
            ['0', '20', '0', '2.07'],
            ['0', '25', '0', '2.06'],
            ['0', '30', '0', '2.07'],
            ['0', '35', '0', '2.07'],
            ['0', '40', '0', '2.07'],
            ['0', '45', '0', '2.07']
        ]

        for i, tar in zip(self.csv1, targets):
            self.assertIsInstance(i, list)
            self.assertEqual(i, tar)

    def test__setitem__(self):
        self.csv1[4] = [1, 2, 3, 6]
        self.assertEqual(self.csv1[4], ['1', '2', '3', '6'])

    def test___len__(self):
        self.assertEqual(len(self.csv1), 10)

    def test___delitem__(self):
        del self.csv1[2]
        self.assertEqual(len(self.csv1), 9)


class TestSession(unittest.TestCase):
    """This test creates a separate 'container' folder to make removing the mess easier."""
    def setUp(self):
        os.mkdir("test_files/container")
        shutil.copytree("test_files/session3", "test_files/container/session3")
        self.session1 = Session(path_to_dir="test_files/container/session3")

    def tearDown(self):
        try:
            shutil.rmtree("test_files/container")
        except FileNotFoundError:
            pass

    def test_add_image(self):
        image = cv2.imread("test_files/session3/0,25,0.jpg")
        self.session1.add_image(img=image, pos=(0, 50, 0))
        self.assertEqual(os.path.isfile("test_files/container/session3/0,50,0.jpg"), True)

    def test_read_images(self):
        model = MyCNN().eval()
        self.session1.read_images(model=model, decimal_pos=1)

        self.assertEqual(self.session1.csv[0][:3], ['0', '0', '0'])
        self.assertEqual(self.session1.csv[1][:3], ['0', '5', '0'])
        self.assertEqual(self.session1.csv[2][:3], ['0', '10', '0'])
        self.assertEqual(self.session1.csv[3][:3], ['0', '15', '0'])
        self.assertEqual(self.session1.csv[4][:3], ['0', '20', '0'])
        self.assertEqual(self.session1.csv[5][:3], ['0', '25', '0'])
        self.assertEqual(self.session1.csv[6][:3], ['0', '30', '0'])
        self.assertEqual(self.session1.csv[7][:3], ['0', '35', '0'])
        self.assertEqual(self.session1.csv[8][:3], ['0', '40', '0'])
        self.assertEqual(self.session1.csv[9][:3], ['0', '45', '0'])

    def test_prepare_for_ml(self):
        result = [
            ('test_files/container/ml_session3', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], []),
            ('test_files/container/ml_session3\\0', [], ['i10n0.jpg', 'i13n0.jpg', 'i16n0.jpg', 'i19n0.jpg',
                                                         'i1n0.jpg', 'i22n0.jpg', 'i25n0.jpg', 'i28n0.jpg',
                                                         'i4n0.jpg', 'i7n0.jpg']),
            ('test_files/container/ml_session3\\1', [], []),
            ('test_files/container/ml_session3\\2', [], ['i0n2.jpg', 'i12n2.jpg', 'i15n2.jpg', 'i18n2.jpg', 'i21n2.jpg',
                                                         'i24n2.jpg', 'i27n2.jpg', 'i3n2.jpg', 'i6n2.jpg', 'i9n2.jpg']),
            ('test_files/container/ml_session3\\3', [], []),
            ('test_files/container/ml_session3\\4', [], []),
            ('test_files/container/ml_session3\\5', [], []),
            ('test_files/container/ml_session3\\6', [], ['i17n6.jpg']),
            ('test_files/container/ml_session3\\7', [], ['i11n7.jpg', 'i14n7.jpg', 'i20n7.jpg', 'i23n7.jpg',
                                                         'i26n7.jpg', 'i29n7.jpg', 'i2n7.jpg', 'i5n7.jpg', 'i8n7.jpg']),
            ('test_files/container/ml_session3\\8', [], []),
            ('test_files/container/ml_session3\\9', [], [])
        ]
        self.session1.prepare_for_ml(target_dir="test_files/container", img_idx=0)
        self.assertEqual(list(os.walk("test_files/container/ml_session3")), result)

    def test_create_empty(self):
        Session.create_empty(path_to_dir="test_files/container")  # creates session4
        self.assertEqual(os.path.isdir("test_files/container/session4"), True)
        self.assertEqual(os.path.isfile("test_files/container/session4/session4.csv"), True)
        self.assertEqual(os.path.isfile("test_files/container/session4/session4.json"), True)

    def test_get_new_session_name(self):
        self.assertEqual(Session.get_new_session_name("test_files"), "session4")

    def test_get_images(self):
        """Attention: the order of the images could be fatal to the test."""
        img_names = [
            "0,0,0.jpg",
            "0,10,0.jpg",
            "0,15,0.jpg",
            "0,20,0.jpg",
            "0,25,0.jpg",
            "0,30,0.jpg",
            "0,35,0.jpg",
            "0,40,0.jpg",
            "0,45,0.jpg",
            "0,5,0.jpg"
        ]
        for i, name in zip(self.session1.images, img_names):
            self.assertIsInstance(i, MyImage)
            self.assertEqual(i.name, name)

        self.assertEqual(len(list(self.session1.images)), len(img_names))


class TestMyCNN(unittest.TestCase):
    def setUp(self):
        self.model = MyCNN()

    def tearDown(self):
        try:
            os.remove("test_files/lcd_cnn_1.pt")
        except FileNotFoundError:
            pass

    def test_save(self):
        self.model.save(name="1", path="test_files")
        self.assertEqual(os.path.isfile("test_files/lcd_cnn_1.pt"), True)

    def test_read(self):
        img = cv2.imread(r"test_files/session3/0,0,0.jpg")
        out = self.model.read(img, decimal_pos=1)
        self.assertIsInstance(out, float)


if __name__ == "__main__":
    unittest.main()
