from math import cos, sin, radians
import matplotlib.pyplot as plt
from rplidar import RPLidar
import matplotlib


matplotlib.use('TkAgg')
plt.ion()

PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(PORT_NAME)
orientation_offset = 90

def process_data(scan, max_distance):
    x_coords = []
    y_coords = []
    colors = []
    left_distances = []
    right_distances = []

    for (_, angle, distance) in scan:
        if distance > 0:
            adjusted_angle = (angle + orientation_offset) % 360
            radians_angle = radians(adjusted_angle)
            x = distance * cos(radians_angle)
            y = distance * sin(radians_angle)
            x_coords.append(x)
            y_coords.append(y)
            colors.append(distance / max_distance)

            if 45 <= adjusted_angle <= 135:
                right_distances.append(distance)
            elif 225 <= adjusted_angle <= 315:
                left_distances.append(distance)

    return x_coords, y_coords, colors, left_distances, right_distances

def analyze_spaces(left_distances, right_distances, threshold=2000):
    left_empty = all(dist > threshold for dist in left_distances) if left_distances else True
    right_empty = all(dist > threshold for dist in right_distances) if right_distances else True
    return left_empty, right_empty

try:
    lidar.start_motor()
    lidar.get_info()
    lidar.get_health()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-6000, 6000)
    ax.set_ylim(-6000, 6000)
    scatter = ax.scatter([], [], c=[], cmap='viridis', s=5)
    max_distance = 5000
    threshold = 2000

    while True:
        for scan in lidar.iter_scans():
            x_coords, y_coords, colors, left_distances, right_distances = process_data(scan, max_distance)
            scatter.set_offsets(list(zip(x_coords, y_coords)))
            scatter.set_array(colors)
            plt.draw()
            plt.pause(0.1)

            left_empty, right_empty = analyze_spaces(left_distances, right_distances, threshold)
            print("Left space is empty." if left_empty else "Left space is not empty.")
            print("Right space is empty." if right_empty else "Right space is not empty.")

finally:
    lidar.stop_motor()
    lidar.stop()
    lidar.disconnect()