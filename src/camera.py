from picamera2 import Picamera2
import cv2
import numpy as np

# Load the MobileNet SSD model
model_path = "MobileNetSSD/"
net = cv2.dnn.readNetFromCaffe(
    model_path + "deploy.prototxt",
    model_path + "mobilenet_iter_73000.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

TARGET_CLASSES = {"person", "bicycle", "motorbike", "car"}

class Camera:
    def __init__(self, camera_num):
        self.camera_num = camera_num
        self.picam = Picamera2(camera_num=camera_num)
        self.picam.configure(self.picam.create_still_configuration())
        self.picam.start()
        print(f"Camera {camera_num} initialized and started.")

    def capture_image(self, save_path=None):
        frame = self.picam.capture_array()
        if save_path:
            cv2.imwrite(save_path, frame)
            print(f"Image captured and saved to {save_path}")
        return frame

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        detected_objects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                if label in TARGET_CLASSES:
                    detected_objects.append(label)

                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    color = (255, 0, 0) if label == "person" else (0, 255, 0)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, text, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for obj in detected_objects:
            print(f"CAMERA {self.camera_num}: {obj.upper()} detected")
        return frame, detected_objects

    def confirm_empty_space(self, frame, save_path=None):
        annotated_frame, detected_objects = self.detect_objects(frame)
        if save_path:
            cv2.imwrite(save_path, annotated_frame)
            print(f"Annotated image saved to {save_path}")
        is_empty = not any(obj in TARGET_CLASSES for obj in detected_objects)
        return is_empty

    def stop(self):
        self.picam.stop()
        print(f"Camera {self.camera_num} stopped.")