import cv2
import json
import numpy
import os
import math
from motors import StepperMotor
import RPi.GPIO as GPIO


class ImageLibrary:  # works
    path: str = "image_library"
    extension: str = ".jpg"
    folder_convention: str = "session"

    def __init__(self):
        # create image_library folder if not existing
        if self.path not in os.listdir():
            os.mkdir(self.path)

        # create unique session-directory name
        directory = os.listdir(self.path)
        largest_num = 0
        for d in directory:
            d = d.replace(self.folder_convention, '')
            if int(d) > largest_num:
                largest_num = int(d)
        self.session_name: str = self.folder_convention + str(largest_num + 1)

        # create directory with the name from above
        self.path = os.path.join(self.path, self.session_name)
        os.mkdir(self.path)
        os.chdir(self.path)

    def append(self, filename: str, image: numpy.array):
        try:
            cv2.imwrite(filename + self.extension, image)
        except Exception:
            print(f"Error saving file ({filename}) as image or text!")


class ParameterHandler:
    def __init__(self, filename: str = "parameters.json"):
        self.default_parameters = {
            "liquid_dimension": (255, 150, 10),
            "step_size": (5, 5, 10),
            "last_pos": (0, 0, 0),
            # "setup": {
            #     "session_name": os.getcwd().split('\\')[-1],
            #     # FIXME: /home/pi/Desktop/equipotential_lines/voltage_mapper
            #     "voltage": input("enter voltage magnitude & AC/DC:"),
            #     "electrode_type": input("enter electrode type:"),
            #     "liquid": input("enter liquid:"),
            # },
        }

        self.filename = filename
        try:
            with open(self.filename) as file:
                try:
                    self.parameters = json.load(file)
                except json.decoder.JSONDecodeError:
                    self.parameters = self.default_parameters
        except Exception:
            self.parameters = self.default_parameters
            self.save_parameters()

        print(f"{self.parameters = }")

    def save_parameters(self):
        with open(self.filename, 'w') as file:
            json.dump(self.parameters, file)


class Arm:
    arm0_height: float = 103
    arm1_length: float = 115.9
    arm2_length: float = 95.9
    arm3_length: float = 80

    def __init__(self, location: tuple[float, float, float], start_pos: tuple[float, float, float] = (0, 0, 0)):
        self.mot_rot = StepperMotor(gpio_pins=[7, 11, 13, 15], reverse=False)
        self.mot_j1 = StepperMotor(gpio_pins=[12, 16, 18, 22], reverse=True)
        self.mot_j2 = StepperMotor(gpio_pins=[19, 21, 23, 29], reverse=False)

        self.location = location
        self.pos = start_pos

        # set the right angles to the
        angles: dict = self.get_angles(pos=start_pos)
        self.mot_rot.set_pos(angle=angles["rot"])
        self.mot_j1.set_pos(angle=angles["j1"])
        self.mot_j2.set_pos(angle=angles["j2"])

    def get_angles(self, pos: tuple[float, float, float]) -> dict:
        """Takes a point (x, y, z) and returns the absolute angles of the motors in the form of a dictionary."""
        def get_angle_triangle(a: float, b: float, c: float, angle: str) -> float:
            """https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html"""
            a, b, c = abs(a), abs(b), abs(c)
            # configuration:
            #   C
            # b/ \a
            # A---B
            #   c
            if angle.lower() == 'a':
                if b + c == a:
                    return 180
                elif b + a == c or c + a == b:
                    return 0
                else:
                    return math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * a)))
            elif angle.lower() == 'b':
                if c + a == b:
                    return 180
                elif c + b == a or a + b == c:
                    return 0
                else:
                    return math.degrees(math.acos((c ** 2 + a ** 2 - b ** 2) / (2 * a * c)))
            elif angle.lower() == 'c':
                if a + b == c:
                    return 180
                elif a + c == b or b + c == a:
                    return 0
                else:
                    return math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
            else:
                raise ValueError("Angle must be 'a', 'b' or 'c'!")
        
        rot = math.degrees(math.atan2(pos[1], pos[0]))  # is: (y, x) for some reason...
        rot = rot + 360 if rot < 0 else rot

        shadow = math.sqrt(pos[0] ** 2 + pos[1] ** 2)  # the "shadow" of the arm on the x-y-plane
        height_diff = pos[2] + self.arm3_length - self.arm0_height  # can be positive and negative
        arm1_and_arm2 = math.sqrt(shadow**2 + height_diff**2)
        origin_j3 = math.sqrt(pos[0]**2 + pos[1]**2 + (pos[2] + self.arm3_length)**2)

        j1 = 180 - get_angle_triangle(a=arm1_and_arm2, b=self.arm0_height, c=origin_j3, angle='C') \
            - get_angle_triangle(a=self.arm2_length, b=self.arm1_length, c=arm1_and_arm2, angle='A')

        j2 = get_angle_triangle(a=self.arm2_length, b=self.arm1_length, c=arm1_and_arm2, angle='C')

        return {"rot": rot, "j1": j1, "j2": j2}

    def move_pos(self, target_pos: tuple[float, float, float]) -> None:
        # calculate angles
        angles: dict = self.get_angles(pos=target_pos)
        self.mot_rot.run_pos(pos=angles["rot"], velocity=.3, hold=False)
        self.mot_j1.run_pos(pos=angles["j1"], velocity=.3, hold=False)
        self.mot_j2.run_pos(pos=angles["j2"], velocity=.3, hold=False)

        self.pos = target_pos


def get_image(camera) -> numpy.array:  # works
    _, image = camera.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # make greyscale
    return image


def main():
    # gpio: [7,11,13,15],[12,16,18,22],[19,21,23,29]
    # 5V Power: 2,4
    # Ground: 6,9
    GPIO.setmode(GPIO.BOARD)

    # cam = cv2.VideoCapture(0)
    # lib = ImageLibrary()  # Changes dir to session folder!
    # param_handler = ParameterHandler()
    arm = Arm(location=(0, 0, 0), start_pos=(0, 211.8, 23))

    # test
    arm.move_pos(target_pos=(14, 6, 2))
    arm.move_pos(target_pos=(0, 211.8, 23))

    # clean everything for next usage
    GPIO.cleanup()


if __name__ == "__main__":
    main()
