from math import pi
import serial
from tqdm import tqdm

from RPi import GPIO as GPIO
from hardware_accessories import StepperMotor, PushButton  # , Camera
from data_accessories import Session, JSON
import ADS1x15


# import torch


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


class Actuator:
    def __init__(self, mot_pins: list, mot_reverse: bool, gear_reduction: float, p_button_pin: int):
        self.mot = StepperMotor(mot_pins, mot_reverse)
        self.gear_reduction = gear_reduction
        self.p_button = PushButton(p_button_pin, self.mot.stop)

    def zero(self):
        """Runs the motor until it hits the push button. The pos is then set to 0."""
        self.p_button.activate()
        self.mot.run(reverse=True)  # is always reverse (not motor dependant)
        self.p_button.deactivate()
        self.mot.pos = 0


class Machine:
    """Class to control the crane-like machine. The machine must be manually moved into its starting position."""

    def __init__(self, json: JSON):
        self.actuator = {
            'x': Actuator(mot_pins=[7, 11, 13, 15], mot_reverse=False, gear_reduction=17 * pi, p_button_pin=38),
            'y': Actuator(mot_pins=[12, 16, 18, 22], mot_reverse=True, gear_reduction=51, p_button_pin=40),
            'z': Actuator(mot_pins=[19, 21, 23, 29], mot_reverse=False, gear_reduction=7.35 * pi / 2, p_button_pin=36)
        }
        self.json = json

    def move_pos(self, new_pos: tuple) -> None:
        """Takes a 3d-position (x,y,z) and moves the measuring tip to that location."""
        for idx, actuator in enumerate(self.actuator.values()):
            distance = new_pos[idx] - self.pos[idx]  # calculate distance
            required_rotations = distance / actuator.gear_reduction  # calculate required rotations in deg
            self.pos[idx] = actuator.mot.run_angle(required_rotations * 360) / 360 * actuator.gear_reduction + self.pos[
                idx]

    def map(self, measuring_function, measuring_kwargs: dict) -> None:
        # plan path
        positions = [[x, y, z]
                     for z in range(0, self.json["area_to_map"][2], self.json["step_size"][2])
                     for y in range(0, self.json["area_to_map"][1], self.json["step_size"][1])
                     for x in range(0, self.json["area_to_map"][0], self.json["step_size"][0])]
        if self.json["pos"] != [0, 0, 0]:
            positions = positions[positions.index(self.json["pos"]):]

        # initialize machine
        self.actuator['x'].mot.run_angle(angle=90, velocity=.5)  # get the motor away from push button
        self.actuator['x'].zero()
        self.actuator['y'].mot.run_angle(angle=90, velocity=.5)
        self.actuator['y'].zero()
        self.actuator['z'].mot.run_angle(angle=90, velocity=.5)
        self.actuator['z'].zero()
        self.pos = (0, 0, 0)
        self.move_pos((0, 0, self.json["distance_to_liquid"]))

        # measure
        for new_pos in tqdm(positions):
            self.move_pos(new_pos)
            measuring_function(pos=new_pos, **measuring_kwargs)
            self.pos = new_pos

            if self.pos[0] + self.json["step_size"][0] == self.json["area_to_map"][0]:
                self.actuator['x'].zero()
                self.pos = (0, self.pos[1], self.pos[2])  # the x pos must be set to 0
            if self.pos[1] + self.json["step_size"][1] == self.json["area_to_map"][1] \
                    and self.pos[0] + self.json["step_size"][0] == self.json["area_to_map"][0]:
                self.actuator['y'].zero()
                self.pos = (self.pos[0], 0, self.pos[2])  # the y pos must be set to 0

        self.move_pos((0, 0, -self.json["distance_to_liquid"]//2))  # raise out of liquid to avoid corrosion

    @property
    def pos(self) -> list:
        """Shortens the access to the position parameter."""
        return self.json["pos"]

    @pos.setter
    def pos(self, new_pos: list or tuple):
        self.json["pos"] = new_pos


def main():
    GPIO.setmode(GPIO.BOARD)

    # define
    active_session = Session()
    machine = Machine(json=active_session.json)
    # camera = Camera()
    # model = torch.load(PATH)
    adc = ADS1x15.ADS1115(1)

    # def optical_measuring(pos):
    #     img = camera.take_picture()
    #     active_session.add_image(img=img, pos=pos)
    #     active_session.csv.append([pos[0], pos[1], pos[2], model.read(img)])

    def electrical_measuring(pos):
        active_session.csv.append([pos[0], pos[1], pos[2], format(adc.toVoltage(adc.readADC(0)), ".2f")])

    if len(active_session.json) == 0:
        active_session.json["area_to_map"] = (90, 160, 50)
        active_session.json["step_size"] = (2, 2, 10)
        active_session.json["pos"] = (0, 0, 0)
        active_session.json["voltage"] = "3V DC"
        active_session.json["electrode_type"] = "2 spheres ~8cm apart"
        active_session.json["liquid"] = "1dm^3 tap water + 5g NaCl"
        active_session.json["distance_to_liquid"] = 19
        active_session.json["liquid_debt"] = 55
        active_session.json["liquid_temp"] = 16

    machine.map(electrical_measuring, {})

    GPIO.cleanup()


if __name__ == "__main__":
    main()
