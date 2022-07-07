#!/usr/bin/env python

"""This file is run from the raspberry pi and takes the measurements as specified in the json file below."""

from math import pi

from tqdm import tqdm
from RPi import GPIO as GPIO
from data_accessories import Session, JSON
import ADS1x15
import torch

from hardware_accessories import StepperMotor, PushButton, Camera
from computervision import MyCNN


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
            'z': Actuator(mot_pins=[19, 21, 23, 29], mot_reverse=True, gear_reduction=7.35 * pi / 2, p_button_pin=36)
        }
        self.json = json

    def move_pos(self, new_pos: tuple) -> None:
        """Takes a 3d-position (x,y,z) and moves the measuring tip to that location."""
        for idx, actuator in enumerate(self.actuator.values()):
            distance = new_pos[idx] - self.pos[idx]  # calculate distance
            required_rotations = distance / actuator.gear_reduction  # calculate required rotations in deg
            self.pos[idx] = actuator.mot.run_angle(required_rotations * 360) / 360 * actuator.gear_reduction \
                + self.pos[idx]

    def map(self, measuring_function, measuring_kwargs: dict) -> None:
        # plan path
        positions = [[x, y, z]
                     for z in range(0, self.json["area_to_map"][2], self.json["step_size"][2])
                     for y in range(0, self.json["area_to_map"][1], self.json["step_size"][1])
                     for x in range(0, self.json["area_to_map"][0], self.json["step_size"][0])]
        if self.json["pos"] != [0, 0, 0]:  # continue measurement if session was not completed
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

        self.move_pos((0, 0, -self.json["distance_to_liquid"] // 2))  # raise out of liquid to avoid corrosion

    @property
    def pos(self) -> list:
        """Shortens the access to the position parameter."""
        return self.json["pos"]

    @pos.setter
    def pos(self, new_pos: list or tuple):
        self.json["pos"] = new_pos


def main():
    GPIO.setmode(GPIO.BOARD)
    session = Session()
    machine = Machine(json=session.json)

    def optical_measuring(pos, cam: Camera, read: bool):
        """
        Records the displayed value of Multimeter display as an image.
        If desired the image can be read immediately
        """
        img = cam.take_picture()
        session.add_image(img=img, pos=pos)
        if read:
            session.csv.append([pos[0], pos[1], pos[2], model.read(img, decimal_pos=1)])

    def electrical_measuring(pos, adc: ADS1x15.ADS1115):
        """The voltage gets recorded via teh ADC."""
        session.csv.append([pos[0], pos[1], pos[2], format(adc.toVoltage(adc.readADC(0)), ".2f")])

    if len(session.json) == 0:
        # All length-measurements are to be entered in mm!
        session.json["area_to_map"] = (0, 0, 0)  # x,y,z limits
        session.json["step_size"] = (0, 0, 0)  # individual step sizes for each dimension
        session.json["pos"] = (0, 0, 0)  # current position (automatically used when interrupted session is used
        session.json["voltage"] = "VOLTAGE"  # specify used voltage for record and to display in plot
        session.json["electrode_type"] = "ELECTRODE_DESCRIPTION"  # describe the electrode setup
        session.json["liquid"] = "LIQUID_DESCRIPTION"  # describe the used liquid
        session.json["distance_to_liquid"] = 0  # distance from measuring tip fully retracted to liquid surface
        session.json["liquid_depth"] = 0
        session.json["liquid_temp"] = 0  # in °C (gets added automatically in the plot)
        session.json["measuring_method"] = "MEASURING_METHOD"  # select "optical" / "electrical" (see code below)
        session.json["model"] = "../models/MODEL_NAME.pt"  # specify model if raspi should read immediately

    if session.json["measuring_method"] == "electrical":
        analog_digital_converter = ADS1x15.ADS1115(1)
        machine.map(electrical_measuring, {"adc": analog_digital_converter})
    elif session.json["measuring_method"] == "optical":
        camera = Camera()
        try:
            model = torch.load(session.json["model"], map_location=torch.device('cpu'))
            model.eval()
            machine.map(optical_measuring, {"cam": camera, "read": True})
        except KeyError:
            machine.map(optical_measuring, {"cam": camera, "read": False})
    else:
        raise ValueError("Invalid measuring method selected!")
    GPIO.cleanup()


if __name__ == "__main__":
    main()
