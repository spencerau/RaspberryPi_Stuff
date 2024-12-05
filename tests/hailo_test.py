from picamera2 import Picamera2
import cv2
import numpy as np
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)


# Model configuration
MODEL_NAME = "yolov5m_wo_spp"
HEF_PATH = f"YOLOv5m_wo_spp/{MODEL_NAME}.hef"
INPUT_RES_H, INPUT_RES_W = 640, 640  # Model input resolution

# Filtered target classes
CLASSES = [
    "person", "bicycle", "car", "motorbike", "bus", "truck"
]
TARGET_CLASSES = {"person", "bicycle", "motorbike", "car", "bus", "truck"}

def preprocess_frame(frame):
    """Resize the frame to 640x640, normalize, and ensure 3 channels."""
    # Convert RGBA to RGB if needed
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Resize to match model input
    resized_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

    # Normalize pixel values to [0, 1]
    normalized_frame = resized_frame / 255.0

    return np.expand_dims(normalized_frame, axis=0).astype(np.float32)  # Shape: (1, 640, 640, 3)

def postprocess_output(output, input_shape, frame):
    """Extract detections from segmentation and draw bounding boxes."""
    output_key = "yolov5m_wo_spp/yolov5_nms_postprocess"

    if output_key not in output:
        print(f"Key '{output_key}' not found in output.")
        return frame

    # Get the data associated with the key
    postprocessed_data = output[output_key]

    # Debugging: Inspect the postprocessed_data
    print(f"Postprocessed Data Type: {type(postprocessed_data)}")
    if isinstance(postprocessed_data, list):
        print(f"Postprocessed Data Length: {len(postprocessed_data)}")

        # If the first element is a list, inspect it further
        if isinstance(postprocessed_data[0], list):
            print("Nested List Detected. Inspecting contents:")
            for idx, sublist in enumerate(postprocessed_data[0]):
                print(f"  Sublist {idx}: Type: {type(sublist)}, Length: {len(sublist) if isinstance(sublist, list) else 'N/A'}")
                if isinstance(sublist, np.ndarray):
                    print(f"    Shape: {sublist.shape}, Example Values: {sublist[:5]}")

        elif isinstance(postprocessed_data[0], np.ndarray):
            segmentation_mask = postprocessed_data[0]
            print(f"Segmentation Mask Shape: {segmentation_mask.shape}")
        else:
            print("Unexpected data type in postprocessed_data[0].")
            return frame
    else:
        print("Postprocessed data is not a list. Cannot process.")
        return frame

    # Assume segmentation_mask is obtained correctly from the previous step
    if not isinstance(segmentation_mask, np.ndarray):
        print("Segmentation mask is not an ndarray. Skipping bounding box generation.")
        return frame

    # Generate bounding boxes from segmentation mask
    h, w = input_shape  # Frame dimensions
    for class_id in np.unique(segmentation_mask):
        if class_id == 0:  # Skip background
            continue

        # Create a binary mask for the current class
        binary_mask = (segmentation_mask == class_id).astype(np.uint8)

        # Find contours or bounding rectangles
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)

            # Add label
            label = f"Class {int(class_id)}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    # Load HEF and configure the Hailo device
    hef = HEF(HEF_PATH)
    devices = Device.scan()  # Scan for available Hailo devices

    with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

        input_shape = hef.get_input_vstream_infos()[0].shape

        # Initialize cameras
        picam2_0 = Picamera2(camera_num=0)
        config0 = picam2_0.create_preview_configuration(main={"size": (1280, 720)})
        picam2_0.configure(config0)
        picam2_0.start()

        picam2_1 = Picamera2(camera_num=1)
        config1 = picam2_1.create_preview_configuration(main={"size": (1280, 720)})
        picam2_1.configure(config1)
        picam2_1.start()

        # OpenCV display windows
        cv2.namedWindow("Camera 0", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera 0", 640, 640)
        cv2.resizeWindow("Camera 1", 640, 640)

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            try:
                while True:
                    # Capture frames from cameras
                    frame0 = picam2_0.capture_array()
                    frame1 = picam2_1.capture_array()

                    # Preprocess frames for inference
                    # input_data_0 = {hef.get_input_vstream_infos()[0].name: preprocess_frame(frame0)}
                    # input_data_1 = {hef.get_input_vstream_infos()[0].name: preprocess_frame(frame1)}

                    # print("Input vstream name:", hef.get_input_vstream_infos()[0].name)
                    # print("Frame 0 shape before preprocessing:", frame0.shape)
                    # print("Frame 1 shape before preprocessing:", frame1.shape)

                    preprocessed_frame0 = preprocess_frame(frame0)
                    preprocessed_frame1 = preprocess_frame(frame1)

                    # print("Preprocessed Frame 0 shape:", preprocessed_frame0.shape)
                    # print("Preprocessed Frame 0 size (bytes):", preprocessed_frame0.nbytes)
                    # print("Preprocessed Frame 1 shape:", preprocessed_frame1.shape)
                    # print("Preprocessed Frame 1 size (bytes):", preprocessed_frame1.nbytes)

                    input_data_0 = {hef.get_input_vstream_infos()[0].name: preprocessed_frame0}
                    input_data_1 = {hef.get_input_vstream_infos()[0].name: preprocessed_frame1}

                    # Run inference
                    with network_group.activate(network_group_params):
                        infer_results_0 = infer_pipeline.infer(input_data_0)
                        infer_results_1 = infer_pipeline.infer(input_data_1)

                    # Postprocess and annotate frames
                    frame0_with_boxes = postprocess_output(infer_results_0, input_shape, frame0)
                    frame1_with_boxes = postprocess_output(infer_results_1, input_shape, frame1)

                    # Display frames
                    cv2.imshow("Camera 0", frame0_with_boxes)
                    cv2.imshow("Camera 1", frame1_with_boxes)

                    # Exit on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                # Stop cameras and close OpenCV windows
                picam2_0.stop()
                picam2_1.stop()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()