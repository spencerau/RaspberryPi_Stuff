from math import cos, sin, pi, floor
import pygame
from adafruit_rplidar import RPLidar

# Set up pygame and the display
pygame.init()
lcd = pygame.display.set_mode((640, 480))  # Adjusted for larger screen
pygame.mouse.set_visible(True)  # Keep mouse visible on macOS
lcd.fill((0, 0, 0))
pygame.display.update()

# Setup the RPLidar
PORT_NAME = '/dev/tty.SLAB_USBtoUART'  # Update for macOS
lidar = RPLidar(None, PORT_NAME)

# Used to scale data to fit on the screen
max_distance = 0

# Process LIDAR data
def process_data(data):
    """Process and plot LiDAR data on the screen."""
    global max_distance
    lcd.fill((0, 0, 0))  # Clear the screen
    for angle in range(360):
        distance = data[angle]
        if distance > 0:  # Ignore invalid data points
            max_distance = max([min([5000, distance]), max_distance])
            radians = angle * pi / 180.0
            x = distance * cos(radians)
            y = distance * sin(radians)
            point = (320 + int(x / max_distance * 239), 240 + int(y / max_distance * 239))
            lcd.set_at(point, pygame.Color(255, 255, 255))  # Draw the point
    pygame.display.update()

# Main program loop
try:
    print("Fetching LiDAR Info...")
    print(lidar.info)  # Print LiDAR information

    scan_data = [0] * 360  # Initialize scan data
    for scan in lidar.iter_scans():
        for (_, angle, distance) in scan:
            scan_data[min([359, floor(angle)])] = distance
        process_data(scan_data)

except KeyboardInterrupt:
    print("Stopping...")
    lidar.stop()
    lidar.disconnect()
    pygame.quit()

except Exception as e:
    print(f"An error occurred: {e}")
    lidar.stop()
    lidar.disconnect()
    pygame.quit()