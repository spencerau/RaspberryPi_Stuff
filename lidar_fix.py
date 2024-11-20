from math import cos, sin, radians
import matplotlib.pyplot as plt
from rplidar import RPLidar

import matplotlib
matplotlib.use('MacOSX')
plt.ion()

# Set up the RPLidar
PORT_NAME = '/dev/tty.SLAB_USBtoUART'
lidar = RPLidar(PORT_NAME)

# Orientation adjustment (in degrees)
# Adjust this to rotate the map to the correct orientation
orientation_offset = 90  # Rotate 90 degrees counterclockwise

def process_data(scan, max_distance):
    """Process LiDAR data and return coordinates with colors for visualization."""
    x_coords = []
    y_coords = []
    colors = []

    for (_, angle, distance) in scan:
        if distance > 0:  # Ignore invalid measurements
            # Apply orientation offset
            adjusted_angle = (angle + orientation_offset) % 360
            radians_angle = radians(adjusted_angle)

            x = distance * cos(radians_angle)
            y = distance * sin(radians_angle)
            x_coords.append(x)
            y_coords.append(y)
            colors.append(distance / max_distance)  # Normalize for color mapping

    return x_coords, y_coords, colors

try:
    # Start the motor
    print("Starting LIDAR motor...")
    lidar.start_motor()

    print("LIDAR Info:")
    print(lidar.get_info())
    print("LIDAR Health:")
    print(lidar.get_health())

    # Set up the plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6000, 6000)
    ax.set_ylim(-6000, 6000)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("RPLidar 2D Map with Orientation")
    scatter = ax.scatter([], [], c=[], cmap='viridis', s=5)

    # Process and visualize the scans
    print("Press Ctrl+C to stop the program.")
    max_distance = 5000  # Set the maximum distance for normalization
    while True:
        for scan in lidar.iter_scans():
            x_coords, y_coords, colors = process_data(scan, max_distance)
            scatter.set_offsets(list(zip(x_coords, y_coords)))
            scatter.set_array(colors)
            plt.draw()
            plt.pause(0.1)  # Update every 0.1 seconds

finally:
    print("Stopping LIDAR motor...")
    lidar.stop_motor()  # Stop the motor after the program ends
    lidar.stop()
    lidar.disconnect()