import json
import csv
import os
import math
import serial
from RPi import GPIO as GPIO
from build.motors import StepperMotor


class BasicHandler:
    """Abstract Class for custom file handlers."""
    def __init__(self, path):
        self.path = path

    def __getnewargs__(self):
        pass

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, item):
        pass


class ParameterHandler(BasicHandler):
    """Ths class organizes the parameters and saves them in a .json file."""
    default_parameters = {
        "area_to_map": (255, 150, 10),
        "step_size": (5, 5, 10),
        "last_pos": (0, 0, 0),
        "session_name": os.getcwd().split('\\')[-1],  # FIXME: /home/pi/Desktop/equipotential_lines/hardware_control
        "voltage": '',
        "electrode_type": '',
        "liquid": '',
    }

    def __init__(self, path: str = "parameters.json"):
        super().__init__(path)

    def get_parameters(self) -> dict:
        try:
            with open(self.path) as file:
                try:
                    parameters = json.load(file)
                except json.decoder.JSONDecodeError:  # if file is empty
                    return self.default_parameters
                return parameters
        except FileNotFoundError:
            return self.default_parameters

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
        """Takes a 3d-position (x,y,z) and moves the measuring tip at that location."""
        for idx, (mot, drivetrain) in enumerate(zip(self.mot.values(), self.drivetrains.values())):
            # calculate distance
            distance = pos[idx] - self.pos[idx]
            required_rotations = distance / drivetrain
            # run
            mot.run_angle(required_rotations*360)
            # set pos
            self.pos[idx] = pos[idx]


def get_folder_name(directory: list, folder_convention: str = "session") -> str:
    """Creates unique directory name."""
    largest_num = 0
    for d in directory:
        d = d.replace(folder_convention, '')
        if int(d) > largest_num:
            largest_num = int(d)
    return folder_convention + str(largest_num + 1)


def main():
    """
    GPIO-pins:     [7,11,13,15],[12,16,18,22],[19,21,23,29]
    5V power:       2,4
    Ground:         6,9
    """
    GPIO.setmode(GPIO.BOARD)

    # create directory
    path = "../sessions"
    if not os.path.isdir(path):
        os.mkdir(path)
    directory = os.listdir(path)
    path = os.path.join(path, get_folder_name(directory))
    os.mkdir(path)
    os.chdir(path)

    # data, parameters & machine
    param_handler = ParameterHandler()
    data_handler = DataHandler()
    machine = Crane()
    master = Master()

    # todo: implement new measuring method

    # prepare GPIO pins for next usage
    GPIO.cleanup()


if __name__ == "__main__":
    main()
