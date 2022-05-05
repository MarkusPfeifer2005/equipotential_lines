import unittest
from build import main


# class TestDigitSorter(unittest.TestCase):
#     def setUp(self) -> None:
#         ...
#
#     def tearDown(self) -> None:
#         ...
#
#     def test_add(self):
#         self.assertEqual(a(1, 2), 3)
#         self.assertEqual(a(-1, 2), 1)
#         self.assertEqual(a(-1, -2), -3)

class TestKinematics(unittest.TestCase):

    def test_get_angle_triangle(self):
        arm = __main__.Arm(location=(0, 0, 0), start_pos=(0, 211.8, 23))
        self.assertEqual(arm.get_angles((0, 211.8, 23)), {"rot": 90, "j1": 0, "j2": 180})



if __name__ == "__main__":
    unittest.main()
