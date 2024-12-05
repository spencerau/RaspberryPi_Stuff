from picamera2 import Picamera2
import cv2

def main():
    # Initialize the first camera (camera_num=0)
    picam2_0 = Picamera2(camera_num=0)
    sensor_size_0 = picam2_0.sensor_resolution

    # Create a custom configuration using the full sensor resolution
    config0 = picam2_0.create_preview_configuration(main={"size": sensor_size_0})
    picam2_0.configure(config0)
    picam2_0.start()
    picam2_0.set_controls({"ScalerCrop": (0, 0, sensor_size_0[0], sensor_size_0[1])})

    # Initialize the second camera (camera_num=1)
    picam2_1 = Picamera2(camera_num=1)
    sensor_size_1 = picam2_1.sensor_resolution

    # Create a custom configuration using the full sensor resolution
    config1 = picam2_1.create_preview_configuration(main={"size": sensor_size_1})
    picam2_1.configure(config1)
    picam2_1.start()
    picam2_1.set_controls({"ScalerCrop": (0, 0, sensor_size_1[0], sensor_size_1[1])})

    # Create OpenCV windows to display the video streams
    cv2.namedWindow("Camera 0", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)

    # Position the windows side by side
    cv2.moveWindow("Camera 0", 100, 100)  # Position for Camera 0
    cv2.moveWindow("Camera 1", 800, 100)  # Position for Camera 1 (offset for side-by-side)

    # Resize the windows
    cv2.resizeWindow("Camera 0", 800, 600)  # Resize Camera 0 window
    cv2.resizeWindow("Camera 1", 800, 600)  # Resize Camera 1 window

    try:
        while True:
            # Capture frames from both cameras
            frame0 = picam2_0.capture_array()
            frame1 = picam2_1.capture_array()

            # Optionally, resize frames for display purposes
            display_frame0 = cv2.resize(frame0, (800, 600))
            display_frame1 = cv2.resize(frame1, (800, 600))

            # Display the frames in their respective windows
            cv2.imshow("Camera 0", display_frame0)
            cv2.imshow("Camera 1", display_frame1)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the cameras and close windows
        picam2_0.stop()
        picam2_1.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()