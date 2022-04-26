import json
import csv
import os
from math import pi as PI
from motors import StepperMotor
import RPi.GPIO as GPIO


class BasicHandler:
    def __init__(self, path):
        self.path = path


class ParameterHandler(BasicHandler):
    default_parameters = {
        "area_to_map": (255, 150, 10),
        "step_size": (5, 5, 10),
        "last_pos": (0, 0, 0),
        "session_name": os.getcwd().split('\\')[-1],  # FIXME: /home/pi/Desktop/equipotential_lines/voltage_mapper
        "voltage": '',
        "electrode_type": '',
        "liquid": '',
    }

    def __init__(self, path: str = "parameters.json"):
        super().__init__(path)

    def get_parameters(self) -> dict:
        with open(self.path) as file:
            try:
                parameters = json.load(file)
            except json.decoder.JSONDecodeError:  # if file is empty
                parameters = self.default_parameters
        return parameters

    def set_parameters(self, **kwargs) -> None:
        # get parameters
        parameters = self.get_parameters()

        # alter parameters
        for key, value in kwargs.items():
            parameters[key] = value

        # save parameters
        self.save_parameters(parameters=parameters)

    def save_parameters(self, parameters):
        """Overrides old version with new parameters."""
        with open(self.path, 'w') as file:
            json.dump(parameters, file)


class DataHandler(BasicHandler):
    delimiter: str = ','

    def __init__(self, path: str = "data.csv"):
        super().__init__(path)

    def append(self, entry: list) -> None:
        with open(self.path, mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=self.delimiter)
            csv_writer.writerow(entry)

    def get_values(self) -> list:
        """Extract values from csv file and returns them in the form of a list."""
        with open(self.path, mode='r') as csv_file:
            return list(csv.reader(csv_file))


class Crane:
    def __init__(self, start_pos: tuple = (0, 0, 0)):
        self.mot = {
            'x': StepperMotor(gpio_pins=[7, 11, 13, 15], reverse=True),
            'y': StepperMotor(gpio_pins=[12, 16, 18, 22], reverse=True),
            'z': StepperMotor(gpio_pins=[19, 21, 23, 29], reverse=False),
        }
        self.drivetrains = {
            'x': 50,
            'y': 10.75 * PI,
            'z': 7.35 * PI / 2,  # div by 2 due to rope configuration un U-shape
        }

        self.pos: list = list(start_pos)

    def move_pos(self, pos: tuple):
        for idx, (mot, drivetrain) in enumerate(zip(self.mot.values(), self.drivetrains.values())):
            # calculate distance
            distance = pos[idx] - self.pos[idx]
            required_rotations = distance / drivetrain
            # run
            mot.run_angle(required_rotations*360)
            # set pos
            self.pos[idx] = pos[idx]


def main():
    # gpio: [7,11,13,15],[12,16,18,22],[19,21,23,29]
    # 5V Power: 2,4
    # Ground: 6,9
    GPIO.setmode(GPIO.BOARD)

    # data, parameters & machine
    param_handler = ParameterHandler()
    data_handler = DataHandler()
    machine = Crane()

    # todo: create session folder
    # todo: delete redundant files
    # todo: implement new measuring method

    # clean everything for next usage
    GPIO.cleanup()


if __name__ == "__main__":
    main()
