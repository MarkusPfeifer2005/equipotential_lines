from machine import UART, Pin, ADC


class LED:
    """
    Simple class for LED handling.
    (+)--->|---(-)
    """
    def __init__(self, pin: int):
        self.led = Pin(pin, Pin.OUT)
        self.is_on = False
    
    def turn_on(self):
        """
        Activates the LED if it is turned off. If it is already turned on
        it remains activated.
        """
        if not self.is_on:
            self.led.toggle()
            self.is_on = True
        
    def turn_off(self):
        """Switches off the LED. If it is already off, it remains deactivated."""
        if self.is_on:
            self.led.toggle()
            self.is_on = False

    def switch_state(self):
        """Changes the state from on to off or off to on!"""
        self.led.toggle()
        if self.is_on:
            self.is_on = False
        else:
            self.is_on = True


class Voltmeter:
    internal_voltage: float = 3.3

    def __init__(self):
        self.conversion_factor = self.internal_voltage / 65535
        self.adc = ADC(Pin(26))

    def get_voltage(self) -> float:
        return self.adc.read_u16() * self.conversion_factor


class Slave:
    """Client instance to communicate with raspberry pi over UART-protocol."""
    def __init__(self):
        self.uart = UART(1, baudrate=9600, tx=Pin(8), rx=Pin(9))
        self.led = LED(pin=25)
        
    def send(self, msg) -> None:
        msg += "\n"
        self.uart.write(msg.encode())

    def receive(self) -> str:
        """Waits until a signal is received and continues writing until '\n' is encountered."""
        while True:
            if self.uart.any():
                return self.uart.read().decode()
