# See: https://www.youtube.com/watch?v=pbCdNh0TiUo for wiring!

from RPi import GPIO as GPIO
import time
import itertools


class DCMotor:
    """Enables control of a basic DC-motor. The motor must be connected to an H-bridge!"""
    def __init__(self, gpio_pins: list[int]):
        self.gpio_pins = gpio_pins

        # Setting modes for pins
        for pin in self.gpio_pins:
            GPIO.setup(pin, GPIO.OUT)

    def run(self, direction: bool, duration: float) -> None:
        """Runs the motor for a given duration in the specified direction."""
        GPIO.output(self.gpio_pins[0], direction)
        GPIO.output(self.gpio_pins[1], not direction)

        time.sleep(duration)

        # Stop the motor
        GPIO.output(self.gpio_pins[0], False)
        GPIO.output(self.gpio_pins[1], False)


class StepperMotor:
    """This class is supposed to control the 28BYJ-48 stepper motor."""
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

    # Hardware-properties of the motor
    teeth_per_layer: int = 8
    teeth_layers: int = 4
    gear_reduction: int = 64

    one_rot: int = teeth_per_layer * teeth_layers * gear_reduction * 2  # in custom degrees

    def __init__(self, gpio_pins: list[int], final_attachment_circumference: float, reverse: bool = False):
        self.seq = list(reversed(self.sequence)) if reverse else self.sequence  # predefining rotation direction

        self.gpio_pins = gpio_pins
        self.final_attachment_circumference_mm = final_attachment_circumference  # in mm

        self.pos: int = 0  # int [0, 4096] | in custom degrees since the motor is restricted to them
        self.last_active_pins = [0, 0, 0, 0]

        # Setting modes for pins
        for pin in self.gpio_pins:
            GPIO.setup(pin, GPIO.OUT)

    def run_angle(self, angle: int) -> None:  # angle in custom degrees
        """Turns the motor a specific angle. Since a stepper motor can only move specific steps, the angle must be
        provided in custom degrees."""
        # set position
        self.set_pos(angle)

        # assembling the sequence
        if self.last_active_pins == [0, 0, 0, 0]:
            seq = self.seq
        else:
            seq = self.seq[self.seq.index(self.last_active_pins):]
            for i in self.seq[:self.seq.index(self.last_active_pins)]:
                seq.append(i)
        # determine the direction the motor is spinning
        if angle < 0:
            seq.reverse()

        try:
            # only work with angles and rotations
            for idx, half_step in enumerate(itertools.cycle(seq)):
                # activate/deactivate pins
                for pin, high_low in zip(self.gpio_pins, half_step):
                    GPIO.output(pin, high_low)
                self.last_active_pins = half_step
                time.sleep(0.001)

                # end the loop if target HAS been reached
                if idx == abs(angle):
                    break

            # deactivate coils (essential to counter overheating)
            for pin in self.gpio_pins:
                GPIO.output(pin, 0)
        except KeyboardInterrupt:  # prevents motor from overheating if process is interrupted
            GPIO.cleanup()

    def run_length(self, length: float) -> None:
        """Uses the run angle method and runs based on a length."""
        required_rotations = length / self.final_attachment_circumference_mm
        custom_degrees = self.one_rot * required_rotations
        cut_degrees = int(custom_degrees)
        self.run_angle(angle=cut_degrees)

    def set_pos(self, angle: int) -> None:  # angle in custom degrees
        # check if angle is int
        if not isinstance(angle, int):
            raise ValueError

        self.pos += angle

        while self.pos > self.one_rot:
            self.pos -= self.one_rot
        while self.pos < 0:
            self.pos += self.one_rot
