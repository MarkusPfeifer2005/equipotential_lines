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
    def __init__(self, location: tuple[float, float, float], start_pos: tuple[float, float, float] = (0, 0, 0)):
        self.mot_rot = StepperMotor(gpio_pins=[7, 11, 13, 15], reverse=True)
        self.mot_j1 = StepperMotor(gpio_pins=[12, 16, 18, 22], reverse=True)
        self.mot_j2 = StepperMotor(gpio_pins=[19, 21, 23, 29], reverse=False)

        self.arm0_height: float = 103.3
        self.arm1_length: float = 115.9
        self.arm2_length: float = 95.9
        self.arm3_length: float = 80

        self.location = location
        self.pos = start_pos

        # set the right angles to the
        angles: dict = self.get_angles(pos=start_pos)
        self.mot_rot.set_pos(angle=angles["rot"])
        self.mot_j1.set_pos(angle=angles["j1"])
        self.mot_j2.set_pos(angle=angles["j2"])

    def get_angles(self, pos: tuple[float, float, float]) -> dict:
        try:
            rot = math.degrees(math.cos(pos[0]))
            j1 = math.degrees(
                math.atan(
                    math.sqrt(  # "shadow"
                        math.sqrt(pos[0] ** 2 + pos[1] ** 2) ** 2 +
                        abs(self.arm0_height - pos[2] + self.arm3_length) ** 2
                    )
                    / abs(self.arm0_height - pos[2]+self.arm3_length)
                )
            )
            j2 = math.degrees(
                math.acos(
                    (
                        self.arm1_length**2 + self.arm2_length**2 -
                        math.sqrt(  # "shadow"
                            math.sqrt(pos[0]**2 + pos[1]**2)**2 +
                            abs(self.arm0_height - pos[2]+self.arm3_length)**2
                        )**2
                    ) / 2*self.arm1_length*self.arm2_length
                )
            )
            return {"rot": rot, "j1": j1, "j2": j2}
        except Exception:
            print(f"target_pos {pos} not in range of arm")
            exit(-1)

    def move_pos(self, target_pos: tuple[float, float, float]) -> None:
        # calculate angles
        angles: dict = self.get_angles(pos=target_pos)
        self.mot_rot.run_pos(pos=angles["rot"], hold=False)
        self.mot_j1.run_pos(pos=angles["j1"], hold=False)
        self.mot_j2.run_pos(pos=angles["j2"], hold=False)

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
    arm = Arm(location=(0, 0, 0), start_pos=(40, 0, 2))

    # test
    arm.move_pos(target_pos=(60, 0, 5))

    # clean everything for next usage
    GPIO.cleanup()


if __name__ == "__main__":
    main()
