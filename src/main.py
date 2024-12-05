from lidar import Lidar
from camera import Camera
import cv2
import os

def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    lidar = Lidar(sensor_model="RPLidar A1")
    empty_spaces = lidar.detect_empty_spaces()

    camera0 = Camera(camera_num=0)
    camera1 = Camera(camera_num=1)

    try:
        for idx, space in enumerate(empty_spaces):
            print(f"Checking space at coordinates {space}...")

            # Capture and save raw images
            frame0_path = os.path.join(output_dir, f"camera0_raw_space_{idx}.jpg")
            frame1_path = os.path.join(output_dir, f"camera1_raw_space_{idx}.jpg")
            frame0 = camera0.capture_image(save_path=frame0_path)
            frame1 = camera1.capture_image(save_path=frame1_path)

            # Confirm empty space and save annotated images
            annotated0_path = os.path.join(output_dir, f"camera0_annotated_space_{idx}.jpg")
            annotated1_path = os.path.join(output_dir, f"camera1_annotated_space_{idx}.jpg")

            is_empty0 = camera0.confirm_empty_space(frame0, save_path=annotated0_path)
            if is_empty0:
                print(f"Space at {space} confirmed empty by Camera 0.")
                continue

            is_empty1 = camera1.confirm_empty_space(frame1, save_path=annotated1_path)
            if is_empty1:
                print(f"Space at {space} confirmed empty by Camera 1.")
            else:
                print(f"Space at {space} is not empty.")

    finally:
        camera0.stop()
        camera1.stop()

if __name__ == "__main__":
    main()