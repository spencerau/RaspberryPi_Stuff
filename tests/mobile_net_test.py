from picamera2 import Picamera2
import cv2
import numpy as np

# Load the MobileNet SSD model
model_path = "MobileNetSSD/"
net = cv2.dnn.readNetFromCaffe(
    model_path + "deploy.prototxt",
    model_path + "mobilenet_iter_73000.caffemodel"
)

# Define the class labels for MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Target classes to detect by their names
TARGET_CLASSES = {"person", "bicycle", "motorbike", "car", "background"}


def detect_objects(frame):
    """Detect specific classes (person, bicycle, motorbike, car, background)."""
    # Ensure the input frame has 3 channels (convert RGBA to RGB if needed)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    height, width = frame.shape[:2]

    # Create a blob and run detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])  # Class index
            label = CLASSES[idx]

            # Focus on specific classes
            if label in TARGET_CLASSES:
                # Extract bounding box
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box
                color = (255, 0, 0) if label == "person" else (0, 255, 0)  # Red for person
                if label == "background":
                    color = (200, 200, 200)  # Grey for background
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                # Add label
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


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
    cv2.moveWindow("Camera 1", 900, 100)  # Position for Camera 1 (offset for side-by-side)

    # Resize the windows
    cv2.resizeWindow("Camera 0", 800, 600)  # Resize Camera 0 window
    cv2.resizeWindow("Camera 1", 800, 600)  # Resize Camera 1 window

    try:
        while True:
            # Capture frames from both cameras
            frame0 = picam2_0.capture_array()
            frame1 = picam2_1.capture_array()

            # Run detection and draw bounding boxes
            frame0_with_boxes = detect_objects(frame0)
            frame1_with_boxes = detect_objects(frame1)

            # Display the frames in their respective windows
            cv2.imshow("Camera 0", frame0_with_boxes)
            cv2.imshow("Camera 1", frame1_with_boxes)

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