import cv2
import json
import numpy
import os
from math import pi as PI
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
    default_parameters = {
        "liquid_dimension": (255, 150, 10),
        "step_size": (5, 5, 10),
        "last_pos": (0, 0, 0),
        "setup": {
            "session_name": os.getcwd().split('\\')[-1],
            "voltage": input("enter voltage magnitude & AC/DC:"),
            "electrode_type": input("enter electrode type:"),
            "liquid": input("enter liquid:"),
        },
    }

    def __init__(self, filename: str = "parameters.json"):
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


def get_image(camera) -> numpy.array:  # works
    _, image = camera.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # make greyscale
    return image


def move_pos(motor: StepperMotor, current_pos: float, target_pos: float):
    """From the voltage_mappers perspective move the motor relative to the current position."""
    travelling_distance = target_pos - current_pos
    if motor.gpio_pins == [19, 21, 23, 29]:  # if the motor is mot_z
        travelling_distance *= 2
    motor.run_length(length=travelling_distance)


def main():
    # gpio: [7,11,13,15],[12,16,18,22],[19,21,23,29]
    # 5V Power: 2,4
    # Ground: 6,9
    GPIO.setmode(GPIO.BOARD)

    # define camera
    cam = cv2.VideoCapture(0)

    # handle parameters and data
    lib = ImageLibrary()  # Changes dir to session folder!
    param_handler = ParameterHandler()

    # define motors
    mot_x = StepperMotor(gpio_pins=[7, 11, 13, 15], final_attachment_circumference=50, reverse=True)
    mot_y = StepperMotor(gpio_pins=[12, 16, 18, 22], final_attachment_circumference=11*PI, reverse=True)
    mot_z = StepperMotor(gpio_pins=[19, 21, 23, 29], final_attachment_circumference=7.5*PI, reverse=False)

    # move to sea level
    difference_sea_level = float(input("enter difference sea-level:"))
    mot_z.run_length(length=difference_sea_level)

    # move and measure
    for z in range(0, param_handler.parameters["liquid_dimension"][2], param_handler.parameters["step_size"][2]):
        move_pos(motor=mot_z, current_pos=param_handler.parameters["last_pos"][2], target_pos=z)
        for x in range(0, param_handler.parameters["liquid_dimension"][0], param_handler.parameters["step_size"][0]):
            move_pos(motor=mot_x, current_pos=param_handler.parameters["last_pos"][0], target_pos=x)
            for y in range(0, param_handler.parameters["liquid_dimension"][1],
                           param_handler.parameters["step_size"][1]):
                move_pos(motor=mot_y, current_pos=param_handler.parameters["last_pos"][1], target_pos=y)

                # measuring
                lib.append(f"x{x}y{y}z{z}", get_image(cam))

                # save state backup
                param_handler.parameters["last_pos"] = (x, y, z)
                param_handler.save_parameters()

    # move to surface
    mot_z.run_length(length=-difference_sea_level)

    # clean everything for next usage
    GPIO.cleanup()


if __name__ == "__main__":
    main()
