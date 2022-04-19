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


def get_angles(self, pos: tuple[float, float, float]) -> dict:
    """Takes a point (x, y, z) and returns the absolute angles of the motors in the form of a dictionary."""

    try:
        # Calculate rot. The rot is given as an absolute position so run_pos must be used!
        rot = math.degrees(math.atan2(pos[1], pos[0]))  # is: (y, x) for some reason...
        rot += 360 if rot < 0 else + 0

        shadow = math.sqrt(pos[0] ** 2 + pos[1] ** 2)  # the "shadow" of the arm on the x-y-plane
        height_diff = pos[2] + self.arm3_length - self.arm0_height  # can be positive and negative
        arm1_and_arm2 = math.sqrt(shadow ** 2 + height_diff ** 2)

        j1 = (
                math.degrees(math.atan(height_diff / shadow))
                +
                get_angle_triangle(a=self.arm2_length, b=self.arm1_length, c=arm1_and_arm2, angle='A')
        )

        j2 = get_angle_triangle(a=self.arm2_length, b=self.arm1_length, c=arm1_and_arm2, angle='C')

        return {"rot": rot, "j1": j1, "j2": j2}
    except Exception:
        print(f"Target position {pos} not in range of arm!")
        exit(-1)


if __name__ == "__main__":
    # unittest.main()

    # test get_angle_triangle
    # print(f"{get_angle_triangle(a=8, b=6, c=7, angle='A') = } should be 77.4° to one decimal place")
    # print(f"{get_angle_triangle(a=8, b=6, c=7, angle='B') = } should be 46.6° to one decimal place")
    # print(f"{get_angle_triangle(a=8, b=6, c=7, angle='C') = } should be 57.9° to one decimal place")

    # test get_angles
    # self.arm0_height: float = 103
    # self.arm1_length: float = 115.9
    # self.arm2_length: float = 95.9
    # self.arm3_length: float = 80

    print(f"{get_angles(self={103, 115.9, 95.9, 80}, pos=(40, 0, 2))}")

