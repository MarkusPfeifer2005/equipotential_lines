import math
import numpy as np
import serial
import cv2
from RPi import GPIO as GPIO
from build.motors import StepperMotor
from data_handling import Session


class Master:
    """Communicates with raspberry pi pico via UART-protocol."""
    def __init__(self, port: str = "/dev/ttyS0", baudrate: int = 9600):
        self.ser = serial.Serial(port=port, baudrate=baudrate)
        print(self.ser)

    def send(self, msg: str) -> None:
        """Sends a message to the client/slave."""
        msg += "\n"
        self.ser.write(msg.encode())

    def receive(self) -> str:
        """Waits until new message is received and returns it as string."""
        return self.ser.read_until().decode()


class Crane:
    """Class to control the crane-like machine. The machine must be manually moved into its starting position."""
    def __init__(self, start_pos: tuple = (0, 0, 0)):
        self.mot = {
            'x': StepperMotor(gpio_pins=[7, 11, 13, 15], reverse=True),
            'y': StepperMotor(gpio_pins=[12, 16, 18, 22], reverse=True),
            'z': StepperMotor(gpio_pins=[19, 21, 23, 29], reverse=False),
        }
        self.drivetrains = {
            'x': 50,
            'y': 10.75 * math.pi,
            'z': 7.35 * math.pi / 2,  # divide by 2 due to rope configuration in U-shape
        }

        self.pos: list = list(start_pos)

    def move_pos(self, pos: tuple) -> None:
        """Takes a 3d-position (x,y,z) and moves the measuring tip to that location."""
        for idx, (mot, drivetrain) in enumerate(zip(self.mot.values(), self.drivetrains.values())):
            # calculate distance
            distance = pos[idx] - self.pos[idx]
            required_rotations = distance / drivetrain
            # run
            mot.run_angle(required_rotations*360)
            # set pos
            self.pos[idx] = pos[idx]


class Camera:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def take_picture(self) -> np.ndarray:
        _, image = self.cam.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


def main():
    """
    GPIO-pins:     [7,11,13,15],[12,16,18,22],[19,21,23,29]
    5V power:       2,4
    Ground:         6,9
    """
    GPIO.setmode(GPIO.BOARD)

    # define
    machine = Crane()
    camera = Camera()
    active_session = Session(path_to_dir="PATH TO DESKTOP")

    active_session.json["area_to_map"] = (255, 150, 10)
    active_session.json["step_size"] = (5, 5, 10)
    active_session.json["last_pos"] = (0, 0, 0)
    active_session.json["voltage"] = ''
    active_session.json["electrode_type"] = ''
    active_session.json["liquid"] = ''

    # measuring takes place from the bottom of the container to the liquid surface
    for z in range(0, active_session.json["area_to_map"][2], active_session.json["step_size"][2]):
        for x in range(0, active_session.json["area_to_map"][0], active_session.json["step_size"][0]):
            for y in range(0, active_session.json["area_to_map"][1], active_session.json["step_size"][1]):
                machine.move_pos((x, y, z))
                active_session.add_image(img=camera.take_picture(), pos=(x, y, z))

    # prepare GPIO pins for next usage
    GPIO.cleanup()


if __name__ == "__main__":
    main()
