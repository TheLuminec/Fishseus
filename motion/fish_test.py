import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

# --- Pin setup ---
MOTORS = {
    "mouth": {"in1": 17, "in2": 27, "en": 22},
    "tail":  {"in1": 23, "in2": 24, "en": 25},
    "body":  {"in1": 5,  "in2": 6,  "en": 12},
}

# Setup pins
for m in MOTORS.values():
    GPIO.setup(m["in1"], GPIO.OUT)
    GPIO.setup(m["in2"], GPIO.OUT)
    GPIO.setup(m["en"], GPIO.OUT)

# Setup PWM (1kHz is safe)
PWMS = {}
for name, m in MOTORS.items():
    pwm = GPIO.PWM(m["en"], 1000)
    pwm.start(0)  # start off
    PWMS[name] = pwm

# --- Motor control helpers ---
def motor_forward(name, speed=60):
    m = MOTORS[name]
    GPIO.output(m["in1"], GPIO.HIGH)
    GPIO.output(m["in2"], GPIO.LOW)
    PWMS[name].ChangeDutyCycle(speed)

def motor_reverse(name, speed=60):
    m = MOTORS[name]
    GPIO.output(m["in1"], GPIO.LOW)
    GPIO.output(m["in2"], GPIO.HIGH)
    PWMS[name].ChangeDutyCycle(speed)

def motor_stop(name):
    m = MOTORS[name]
    GPIO.output(m["in1"], GPIO.LOW)
    GPIO.output(m["in2"], GPIO.LOW)
    PWMS[name].ChangeDutyCycle(0)

def stop_all():
    for name in MOTORS:
        motor_stop(name)

# --- Wiggle Test ---
try:
    print("Starting wiggle test...")

    while True:
        # Tail wiggle
        motor_forward("tail", 60)
        time.sleep(0.2)
        motor_reverse("tail", 80)
        time.sleep(0.2)
        motor_stop("tail")

        # Body flop
        motor_forward("body", 70)
        time.sleep(0.25)
        motor_reverse("body", 60)
        time.sleep(0.20)
        motor_stop("body")

        # Mouth snap
        motor_forward("mouth", 80)
        time.sleep(0.15)
        motor_reverse("mouth", 20)
        time.sleep(0.10)
        motor_stop("mouth")

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stop_all()
    GPIO.cleanup()
