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
        # create unique directory name
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
            try:
                open(filename + ".txt", 'w')
            except Exception:
                print(f"Error saving file ({filename}) as image or text!")


class ParameterHandler:
    default_parameters = {
        "container_x_mm": 255,
        "container_y_mm": 150,
        "last_pos": (0, 0),
        "last_session_name": os.getcwd().split('\\')[-1],
        "step_mm": 5
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
    motor.run_length(length=travelling_distance)


def main():
    # gpio: [7,11,13,15],[12,16,18,22]
    # 5V Power: 2,4
    # Ground: 6,9
    GPIO.setmode(GPIO.BOARD)

    mot_x = StepperMotor(gpio_pins=[7, 11, 13, 15], final_attachment_circumference=50, reverse=True)
    mot_y = StepperMotor(gpio_pins=[12, 16, 18, 22], final_attachment_circumference=11*PI, reverse=True)

    cam = cv2.VideoCapture(0)

    lib = ImageLibrary()  # Changes dir to session folder!
    param_handler = ParameterHandler()

    for x in range(0, param_handler.parameters["container_x_mm"], param_handler.parameters["step_mm"]):
        move_pos(motor=mot_x, current_pos=param_handler.parameters["last_pos"][0], target_pos=x)
        for y in range(0, param_handler.parameters["container_y_mm"], param_handler.parameters["step_mm"]):
            move_pos(motor=mot_y, current_pos=param_handler.parameters["last_pos"][1], target_pos=y)

            # measuring
            # time.sleep(2)
            lib.append(f"x{x}y{y}", get_image(cam))

            # save state backup
            param_handler.parameters["last_pos"] = (x, y)
            param_handler.save_parameters()

    GPIO.cleanup()


if __name__ == "__main__":
    main()
