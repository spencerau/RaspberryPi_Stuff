from picamera2 import Picamera2
import cv2
import numpy as np


model_path = "MobileNetSSD/"
net = cv2.dnn.readNetFromCaffe(
    model_path + "deploy.prototxt",
    model_path + "mobilenet_iter_73000.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

TARGET_CLASSES = {"person", "background" "bicycle", "motorbike", "car"}

def detect_objects(frame, camera_id):
    """Detect specific classes and draw bounding boxes."""
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
        print(f"CAMERA {camera_id}: {obj.upper()} detected")

    return frame

def main():
    picam2_0 = Picamera2(camera_num=0)
    picam2_1 = Picamera2(camera_num=1)

    picam2_0.configure(picam2_0.create_still_configuration())
    picam2_1.configure(picam2_1.create_still_configuration())

    picam2_0.start()
    picam2_1.start()

    try:
        frame0 = picam2_0.capture_array()
        frame1 = picam2_1.capture_array()

        frame0_with_boxes = detect_objects(frame0, 0)
        frame1_with_boxes = detect_objects(frame1, 1)

        cv2.imwrite("output/camera0_output.jpg", frame0_with_boxes)
        cv2.imwrite("output/camera1_output.jpg", frame1_with_boxes)

        print("Images saved as camera0_output.jpg and camera1_output.jpg")

        cv2.imshow("Camera 0", frame0_with_boxes)
        cv2.imshow("Camera 1", frame1_with_boxes)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        picam2_0.stop()
        picam2_1.stop()

if __name__ == "__main__":
    main()