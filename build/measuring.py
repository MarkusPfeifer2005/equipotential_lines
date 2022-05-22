import math
import numpy as np
import serial
import cv2
from RPi import GPIO as GPIO
from hardware_accessories import StepperMotor, PushButton
from data_accessories import RpiSession


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


class Machine:
    """Class to control the crane-like machine. The machine must be manually moved into its starting position."""
    def __init__(self, start_pos: tuple = (0, 0, 0)):
        self.mot = {
            'x': StepperMotor(gpio_pins=[7, 11, 13, 15], reverse=False),
            'y': StepperMotor(gpio_pins=[12, 16, 18, 22], reverse=True),
            'z': StepperMotor(gpio_pins=[19, 21, 23, 29], reverse=False),
        }
        self.drivetrains = {
            'x': 53.3,  # estimated value (calculated: 17pi)
            'y': 51,  # measured value
            'z': 7.35 * math.pi / 2,  # divide by 2 due to rope configuration in U-shape
        }
        self.p_button = PushButton(pin=37, function=self.mot['x'].stop)
        self.pos: list = list(start_pos)

    def zero_x(self):
        """Runs the motor x until it hits the push button. The x pos is then set to 0."""
        self.p_button.activate()
        self.mot['x'].run(reverse=True)
        self.p_button.deactivate()
        self.pos[0] = 0

    def move_pos(self, pos: tuple) -> None:
        """Takes a 3d-position (x,y,z) and moves the measuring tip to that location."""
        for idx, (mot, drivetrain) in enumerate(zip(self.mot.values(), self.drivetrains.values())):
            distance = pos[idx] - self.pos[idx]  # calculate distance
            required_rotations = distance / drivetrain  # calculate required rotations of the motor in deg
            self.pos[idx] = mot.run_angle(required_rotations*360) / 360 * drivetrain + self.pos[idx]  # set pos & run


class Camera:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def take_picture(self) -> np.ndarray:
        _, image = self.cam.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


def main():
    GPIO.setmode(GPIO.BOARD)

    # define
    machine = Machine()
    camera = Camera()
    active_session = RpiSession()

    if len(active_session.json) == 0:
        active_session.json["area_to_map"] = (125, 225, 25)
        active_session.json["step_size"] = (5, 5, 5)
        active_session.json["last_pos"] = (0, 0, 0)
        active_session.json["voltage"] = "12V AC"
        active_session.json["electrode_type"] = "2 lines 10cm from each other apart"
        active_session.json["liquid"] = "dusty 1 day old tap water"
        active_session.json["ground_clearance"] = 5
        active_session.json["liquid_debt"] = 32
        active_session.json["liquid_temp"] = 18

    machine.mot['x'].run_angle(angle=180, velocity=.5)  # get the motor away from push button
    machine.zero_x()
    for z in range(0, active_session.json["area_to_map"][2], active_session.json["step_size"][2]):
        for y in range(0, active_session.json["area_to_map"][1], active_session.json["step_size"][1]):
            for x in range(0, active_session.json["area_to_map"][0], active_session.json["step_size"][0]):
                machine.move_pos((x, y, z))
                active_session.add_image(img=camera.take_picture(), pos=(x, y, z))
            machine.zero_x()

    GPIO.cleanup()  # prepare GPIO pins for next usage


if __name__ == "__main__":
    main()
