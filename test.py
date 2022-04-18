import unittest
import math


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

# class TestKinematics(unittest.TestCase):
#
#     def test_get_angle_triangle(self):
#         arm = equipotential_lines.__main__.Arm(location=(0, 0, 0), start_pos=(0, 211.8, 23))
#         self.assertEqual(arm.get_angles((0, 211.8, 23)), {"rot": 90, "j1": 0, "j2": 180})


def get_angle_triangle(a: float, b: float, c: float, angle: str) -> float:
    """https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html"""
    # configuration:
    #   C
    # a/ \b
    # A---B
    #   c
    if angle.lower() == 'a':
        return math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * a)))
    elif angle.lower() == 'b':
        return math.degrees(math.acos((c ** 2 + a ** 2 - b ** 2) / (2 * a * c)))
    elif angle.lower() == 'c':
        return math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
    else:
        raise ValueError("Angle must be 'a', 'b' or 'c'!")


if __name__ == "__main__":
    # unittest.main()

    print(f"{get_angle_triangle(a=8, b=6, c=7, angle='A')} should be 77.4° to one decimal place")
    print(f"{get_angle_triangle(a=8, b=6, c=7, angle='B')} should be 46.6° to one decimal place")
    print(f"{get_angle_triangle(a=8, b=6, c=7, angle='C')} should be 57.9° to one decimal place")
