"""
Hardware specific parts of the machine get controlled from here. Further, plans involve to make an extra repository
where such things like motors get controlled with c++ and a python API, but no concrete schedule is provided!
"""

from RPi import GPIO as GPIO
import time
import itertools
import cv2
import numpy as np


class DCMotor:
    """Controls a basic DC-motor. The motor must be connected to an H-bridge!"""
    def __init__(self, gpio_pins: list[int]):
        self.gpio_pins = gpio_pins

        # Setting modes for pins
        for pin in self.gpio_pins:
            GPIO.setup(pin, GPIO.OUT)

    def run(self, direction: bool, duration: float):
        """Runs the motor for a given duration in the specified direction."""
        GPIO.output(self.gpio_pins[0], direction)
        GPIO.output(self.gpio_pins[1], not direction)

        time.sleep(duration)

        # Stop the motor
        GPIO.output(self.gpio_pins[0], False)
        GPIO.output(self.gpio_pins[1], False)


class StepperMotor(DCMotor):
    """
    Controls the 28BYJ-48 stepper motor.
    Attention: Since a stepper motor can only move steps it can not get the exact position. But past errors are
    compensated for the next move, so they do not sum up. The error in positioning is 1 step at maximum.
    """
    sequence: list = [
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
        super(StepperMotor, self).__init__(gpio_pins)
        self.seq = list(reversed(self.sequence)) if reverse else self.sequence  # predefining rotation direction

        self._pos: int = 0  # given in steps
        self.last_active_pins = self.seq[0]
        self.is_running = False  # only used for the run-method

    def _run_steps(self, steps: int, velocity: float = 1, hold: bool = False):
        """Turns the motor a given amount of steps."""
        # set position
        self._pos = (self._pos + steps) % self.steps_per_rot
        # assembling the sequence
        seq = self.seq[self.seq.index(self.last_active_pins):]
        for i in self.seq[:self.seq.index(self.last_active_pins)]:
            seq.append(i)
        # determine the direction the motor is spinning
        if steps < 0:
            seq.reverse()
        # rotate the motor
        try:
            for idx, step in enumerate(itertools.cycle(seq)):
                # activate/deactivate pins
                for pin, high_low in zip(self.gpio_pins, step):
                    GPIO.output(pin, high_low)
                self.last_active_pins = step
                time.sleep(0.001 / velocity)

                # end the loop if target has been reached
                if idx == abs(steps):
                    break

        except KeyboardInterrupt:  # prevents motor from overheating if process is interrupted
            GPIO.cleanup()
        finally:
            # deactivate coils (essential to counter overheating)
            if not hold:
                for pin in self.gpio_pins:
                    GPIO.output(pin, 0)

    def run_angle(self, angle: float, velocity: float = 1, hold: bool = False) -> float:
        """
        Turns the motor a specific angle. Since only a certain number of steps is possible it reaches to the
        nearest position possible. The inaccuracies do not sum up!
        """
        steps: int = round(self.steps_per_rot * angle / 360)  # calculate the required steps
        self._run_steps(steps=steps, velocity=velocity, hold=hold)  # rotate the motor for the calculated steps
        return steps / self.steps_per_rot * 360  # returns the actual - not the desired position in degrees

    def run_pos(self, pos: int, velocity: float = 1, hold: bool = False):
        """
        Runs the motor to a given position. The velocity is set via a growth-factor (1 is the full speed).
        If hold is false, the motor is allowed to spin freely.
        """
        angle = pos - self.pos
        self.run_angle(angle=angle, velocity=velocity, hold=hold)

    @property
    def pos(self) -> float:
        """Returns the degree equivalent to the real position."""
        return self._pos / self.steps_per_rot * 360

    @pos.setter
    def pos(self, new_angle: float):
        """Takes the new angel in degrees and calculates the closest position/step to the new angle."""
        self._pos = round(self.steps_per_rot * new_angle / 360)

    def run(self, reverse: bool, velocity: float = 1):
        """Not implemented yet."""
        # assembling the sequence
        seq = self.seq[self.seq.index(self.last_active_pins):]
        for i in self.seq[:self.seq.index(self.last_active_pins)]:
            seq.append(i)
        # determine the direction the motor is spinning
        if reverse:
            seq.reverse()
        # rotate the motor
        self.is_running = True
        try:
            for idx, step in enumerate(itertools.cycle(seq)):
                # activate/deactivate pins
                for pin, high_low in zip(self.gpio_pins, step):
                    GPIO.output(pin, high_low)
                self.last_active_pins = step
                time.sleep(0.001 / velocity)

                # end the loop if requested
                if not self.is_running:
                    break

        except KeyboardInterrupt:  # prevents motor from overheating if process is interrupted
            GPIO.cleanup()
        finally:
            # deactivate coils (essential to counter overheating)
            for pin in self.gpio_pins:
                GPIO.output(pin, 0)

    def stop(self):
        self.is_running = False


class PushButton:
    """The Push button possesses a function that is called whenever the button is pushed (and released?)."""

    def __init__(self, pin: int, function, is_active: bool = False):
        self.function = function
        self.pin = pin
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        if is_active:
            self.activate()

    def task(self, channel):  # channel is necessary because something is passed in
        self.function()

    def activate(self):
        GPIO.add_event_detect(self.pin, GPIO.RISING, callback=self.task)

    def deactivate(self):
        GPIO.remove_event_detect(self.pin)


class Camera:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def take_picture(self) -> np.ndarray:
        _, image = self.cam.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # reconvert to rgb for cnn (gets converted when saving)
        return image
