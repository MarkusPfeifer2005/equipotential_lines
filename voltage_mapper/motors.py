# See: https://www.youtube.com/watch?v=pbCdNh0TiUo for wiring!

from RPi import GPIO as GPIO
import time
import itertools
from math import pi as PI


class DCMotor:
    def __init__(self, gpio_pins: list[int]):
        self.gpio_pins = gpio_pins

        # Setting modes for pins
        for pin in self.gpio_pins:
            GPIO.setup(pin, GPIO.OUT)

    def run(self, direction: bool, duration: float):
        GPIO.output(self.gpio_pins[0], direction)
        GPIO.output(self.gpio_pins[1], not direction)

        time.sleep(duration)

        GPIO.output(self.gpio_pins[0], False)
        GPIO.output(self.gpio_pins[1], False)


class StepperMotor:
    sequence = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]

    # hardware-properties of the motor
    teeth_per_layer: int = 8
    teeth_layers: int = 4
    gear_reduction: int = 64

    one_rot_cst_deg: int = teeth_per_layer * teeth_layers * gear_reduction * 2

    def __init__(self, gpio_pins: list[int], final_attachment_circumference_mm: float, reverse: bool = False):
        self.seq = list(reversed(self.sequence)) if reverse else self.sequence  # predefining rotation direction

        self.gpio_pins = gpio_pins
        self.final_attachment_circumference_mm = final_attachment_circumference_mm  # in mm

        self.pos_cst_deg: int = 0  # int [0, 4096]
        self.active_pins = [0, 0, 0, 0]

        # Setting modes for pins
        for pin in self.gpio_pins:
            GPIO.setup(pin, GPIO.OUT)

    def run_angle(self, angle_cst_deg: int):
        # set position
        self.set_pos(angle_cst_deg)

        # assembling the sequence
        if self.active_pins == [0, 0, 0, 0]:
            seq = self.seq
        else:
            seq = self.seq[self.seq.index(self.active_pins):]
            for i in self.seq[:self.seq.index(self.active_pins)]:
                seq.append(i)
        # determine the direction the motor is spinning
        if angle_cst_deg < 0:
            seq.reverse()

        # only work with angles and rotations
        for idx, half_step in enumerate(itertools.cycle(seq)):
            # activate/deactivate pins
            for pin, high_low in zip(self.gpio_pins, half_step):
                GPIO.output(pin, high_low)
            self.active_pins = half_step
            time.sleep(0.001)

            # end the loop if target HAS been reached
            if idx == abs(angle_cst_deg):
                break

    def run_length(self, length: float):
        required_rotations = length / self.final_attachment_circumference_mm
        custom_degrees = self.one_rot_cst_deg * required_rotations
        cut_degrees = int(custom_degrees)
        self.run_angle(angle_cst_deg=cut_degrees)

    def set_pos(self, angle_cst_deg: int):
        # check if angle is int
        if not isinstance(angle_cst_deg, int):
            raise ValueError

        self.pos_cst_deg += angle_cst_deg

        while self.pos_cst_deg > self.one_rot_cst_deg:
            self.pos_cst_deg -= self.one_rot_cst_deg
        while self.pos_cst_deg < 0:
            self.pos_cst_deg += self.one_rot_cst_deg


def main():
    # gpio: [7,11,13,15],[12,16,18,22]
    # 5V Power: 2,4
    # Ground: 6,9
    GPIO.setmode(GPIO.BOARD)

    mot1 = StepperMotor(gpio_pins=[12, 16, 18, 22], final_attachment_circumference_mm=11*PI)
    mot2 = StepperMotor(gpio_pins=[7, 11, 13, 15], final_attachment_circumference_mm=50)

    # mot1.run_length(length=50)
    mot2.run_length(length=50)

    GPIO.cleanup()


if __name__ == "__main__":
    main()
