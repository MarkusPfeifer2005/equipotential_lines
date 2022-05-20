from RPi import GPIO as GPIO
import time
import itertools


class DCMotor:
    """Controls a basic DC-motor. The motor must be connected to an H-bridge!"""
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


class StepperMotor(DCMotor):
    """
    Controls the 28BYJ-48 stepper motor.
    Attention: there is a slight error in the positioning system, that causes errors while positioning. I assume these
    errors originate from the round function, that is responsible for some minimal inaccuracies, that only show up
    when travelling longer distances!
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

        self._pos: int = 0  # value between 0 and 360
        self.last_active_pins = self.seq[0]

    def _run_steps(self, steps: int, velocity: float = 1, hold: bool = False) -> None:
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

            # deactivate coils (essential to counter overheating)
            if not hold:
                for pin in self.gpio_pins:
                    GPIO.output(pin, 0)
        except KeyboardInterrupt:  # prevents motor from overheating if process is interrupted
            GPIO.cleanup()

    def run_angle(self, angle: float, velocity: float = 1, hold: bool = False) -> float:
        """
        Turns the motor a specific angle. Since only a certain number of steps is possible it reaches to the
        nearest position possible. The inaccuracies do not sum up!
        """
        steps: int = round(self.steps_per_rot * angle / 360)
        self._run_steps(steps=steps, velocity=velocity, hold=hold)
        return steps / self.steps_per_rot * 360

    def run_pos(self, pos: int, velocity: float = 1, hold: bool = False) -> None:
        """
        Runs the motor to a given position. The velocity is set via a growth-factor (1 is the full speed).
        If hold is false, the motor is allowed to spin freely.
        """
        angle = pos - self.pos
        self.run_angle(angle=angle, velocity=velocity, hold=hold)

    @property
    def pos(self):
        """Returns the degree equivalent to the real position."""
        return self._pos / self.steps_per_rot * 360

    def run(self, direction: bool, duration: float) -> None:
        """Not implemented yet."""
        pass


class PushButton:
    """The function is put into a wrapper method an is only executed if the button is activated."""
    def __init__(self, pin: int, function, is_active: bool = False):
        self.function = function
        self._is_active = is_active
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(pin, GPIO.RISING, callback=self.task)

    def task(self):
        if self._is_active:
            self.function()

    def activate(self):
        self._is_active = True

    def deactivate(self):
        self._is_active = False

    def toggle(self):
        self._is_active = False if self._is_active else True
