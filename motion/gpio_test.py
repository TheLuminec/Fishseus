import gpiozero as gpio
from time import sleep

led = gpio.LED(17)

while True:
    led.on()
    sleep(1)
    led.off()
    sleep(1)