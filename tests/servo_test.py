from gpiozero import Servo
from time import sleep

# Set up the servo
servo = Servo(21, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)  # GPIO2

def angle_to_position(angle):
    """
    Convert an angle (0 to 180 degrees) to a normalized position (-1 to 1).
    """
    return (angle / 90) - 1

try:
    while True:
        angle = float(input("Enter angle (0-180): "))
        if 0 <= angle <= 180:
            position = angle_to_position(angle)
            servo.value = position
            print(f"Moved to {angle} degrees (position {position})")
            sleep(1)
        else:
            print("Angle must be between 0 and 180 degrees!")
except KeyboardInterrupt:
    print("Exiting...")
