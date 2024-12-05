import os
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

TARGET_CLASSES = {"car", "bicycle", "motorbike", "person"}

def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            if label in TARGET_CLASSES:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, "CAR", (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return frame

def detect_disabled_parking(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if aspect_ratio > 2:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "DISABLED_PARKING", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    return frame

def process_images(input_dir):
    if not os.path.isdir(input_dir):
        return
    output_dir = f"{input_dir.rstrip('/')}_detect"
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        frame = cv2.imread(input_path)
        if frame is None:
            continue
        frame = detect_objects(frame)
        frame = detect_disabled_parking(frame)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)

if __name__ == "__main__":
    input_subdir = "parking"
    process_images(input_subdir)