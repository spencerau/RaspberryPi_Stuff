from picamera2 import Picamera2
import cv2
import numpy as np
from hailo_platform import HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams

# Hailo model path
HEF_PATH = "MobileNetSSD/ssd_mobilenet_v1.hef"

# Class labels for MobileNetSSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Target classes to detect
TARGET_CLASSES = {"person", "bicycle", "motorbike", "car", "bus", "truck"}


def preprocess_frame(frame, input_shape):
    """Preprocess frame for Hailo inference."""
    # Ensure the frame has exactly 3 channels (convert RGBA to RGB if needed)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Resize the frame to the expected input shape
    resized_frame = cv2.resize(frame, input_shape)

    # Clip values to [0, 255] and convert to uint8
    processed_frame = np.clip(resized_frame, 0, 255).astype(np.uint8)

    return processed_frame


def postprocess_output(output, frame, width, height):
    """Postprocess Hailo inference output and draw bounding boxes."""
    detections = output.get("ssd_mobilenet_v1/nms1", [])

    # Debugging: Inspect the detection output structure
    print(f"Detection Output Type: {type(detections)}")
    print(f"Detection Output Length: {len(detections)}")
    for i, sublist in enumerate(detections):
        print(f"Detection Sublist {i}: Type: {type(sublist)}, Length: {len(sublist) if isinstance(sublist, list) else 'N/A'}")
        if isinstance(sublist, list) and len(sublist) > 0:
            print(f"Sample Detection {i}: {sublist[0]}")

    if not isinstance(detections, list):
        print("Unexpected detections format: Not a list.")
        return frame

    total_valid_detections = 0
    for sublist in detections:
        if not isinstance(sublist, list):
            print(f"Unexpected sublist format: {sublist}")
            continue

        for detection in sublist:
            if len(detection) != 5:
                print(f"Unexpected detection format: {detection}")
                continue

            xmin, ymin, xmax, ymax, confidence = detection

            # Debug raw detection values
            print(f"Detection Raw Values - xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}, confidence: {confidence}")

            if confidence > 0.5:  # Adjust threshold as needed
                # Convert normalized coordinates to pixel values
                startX = int(xmin * width)
                startY = int(ymin * height)
                endX = int(xmax * width)
                endY = int(ymax * height)

                # Clamp values to frame boundaries
                startX = max(0, min(startX, width - 1))
                startY = max(0, min(startY, height - 1))
                endX = max(0, min(endX, width - 1))
                endY = max(0, min(endY, height - 1))

                # Debug scaled values
                print(f"Scaled Coordinates - startX: {startX}, startY: {startY}, endX: {endX}, endY: {endY}")

                # Draw bounding box
                color = (0, 255, 0)  # Green for bounding boxes
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                # Add label
                label = f"Confidence: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                total_valid_detections += 1

    print(f"Total valid detections: {total_valid_detections}")
    return frame


def main():
    # Load the Hailo model
    hef = HEF(HEF_PATH)
    devices = Device.scan()
    if not devices:
        raise RuntimeError("No Hailo devices found.")

    with VDevice(device_ids=devices) as target:
        # Configure the Hailo device
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        input_vstream_info = hef.get_input_vstream_infos()[0]
        input_shape = (int(input_vstream_info.shape[1]), int(input_vstream_info.shape[2]))  # H, W

        # Create input and output vstream parameters
        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group)

        # Initialize Camera 0 only
        picam2_0 = Picamera2(camera_num=0)
        sensor_size = picam2_0.sensor_resolution
        config = picam2_0.create_preview_configuration(main={"size": sensor_size})
        picam2_0.configure(config)
        picam2_0.start()
        picam2_0.set_controls({"ScalerCrop": (0, 0, sensor_size[0], sensor_size[1])})

        # Create an OpenCV window
        cv2.namedWindow("Camera 0", cv2.WINDOW_NORMAL)

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                try:
                    while True:
                        # Capture frame from Camera 0
                        frame0 = picam2_0.capture_array()

                        # Get frame dimensions
                        height, width = frame0.shape[:2]
                        input_shape = (300, 300)

                        # Preprocess and infer
                        processed_frame = preprocess_frame(frame0, input_shape)
                        #print(f"Processed Frame Shape: {processed_frame.shape}, Data Type: {processed_frame.dtype}")
                        input_data_0 = {input_vstream_info.name: processed_frame.reshape(1, *input_shape, 3)}
                        infer_results_0 = infer_pipeline.infer(input_data_0)

                        # Debugging inference output
                        #print(f"Infer Results Keys: {list(infer_results_0.keys())}")
                        #print(f"Infer Results Sample: {infer_results_0}")

                        # Postprocess and draw bounding boxes
                        frame0_with_boxes = postprocess_output(infer_results_0, frame0, width, height)

                        # Display frame
                        cv2.imshow("Camera 0", frame0_with_boxes)

                        # Exit on 'q' key press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                finally:
                    # Cleanup
                    picam2_0.stop()
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()