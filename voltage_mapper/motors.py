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

    steps_per_rot: int = teeth_per_layer * teeth_layers * gear_reduction * 2

    def __init__(self, gpio_pins: list[int], reverse: bool = False):
        self.seq = list(reversed(self.sequence)) if reverse else self.sequence  # predefining rotation direction

        self.gpio_pins = gpio_pins

        self.pos: float = 0  # value between 0 and 360
        self.last_active_pins = [0, 0, 0, 0]

        # Setting modes for pins
        for pin in self.gpio_pins:
            GPIO.setup(pin, GPIO.OUT)

    def run_angle(self, angle: float, velocity: float = 1, hold: bool = False) -> None:  # angle in custom degrees
        """Turns the motor a specific angle."""
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

        # rotate the motor
        try:
            for idx, half_step in enumerate(itertools.cycle(seq)):
                # activate/deactivate pins
                for pin, high_low in zip(self.gpio_pins, half_step):
                    GPIO.output(pin, high_low)
                self.last_active_pins = half_step
                time.sleep(0.001 / velocity)

                # end the loop if target HAS been reached
                # get the closest step to desired position
                #   360   :   4096
                #   angle :   angle in steps
                angle_in_steps = round(self.steps_per_rot * abs(angle) / 360)

                if idx == angle_in_steps:
                    break

            # deactivate coils (essential to counter overheating)
            if not hold:
                for pin in self.gpio_pins:
                    GPIO.output(pin, 0)
        except KeyboardInterrupt:  # prevents motor from overheating if process is interrupted
            GPIO.cleanup()

    def run_pos(self, pos: int, velocity: float = 1, hold: bool = False) -> None:
        """Runs the motor to a given position."""
        angle = pos - self.pos
        self.run_angle(angle=angle, velocity=velocity, hold=hold)

    def set_pos(self, angle: float) -> None:  # angle in custom degrees
        """Sets the position the motor is at."""
        self.pos += angle

        while self.pos > 360:
            self.pos -= 360
        while self.pos < 0:
            self.pos += 360
