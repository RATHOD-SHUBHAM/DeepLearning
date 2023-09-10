# todo: Import the model functions and classes from the SAHI directory.
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

# todo: Register the model
yolov8_model_path = "weights/yolov8x.pt"

# todo: Standard Inference with a YOLOv8 Model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.25,
    device="cpu", # or 'cuda:0'
)


input_path = 'inputImage/flockOfbirds.jpg'

# todo: Inference without SAHI
result = get_prediction(input_path, detection_model)

result.export_visuals(export_dir="outputImage/")

# todo: SAHI + YoloV8

result = get_sliced_prediction(
    input_path,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

# Make prediction
import cv2
from numpy import asarray
from sahi.prediction import visualize_object_predictions


img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
numpydata = asarray(img_converted)

visualize_object_predictions(
    numpydata,
    object_prediction_list = result.object_prediction_list,
    hide_labels = 0,
    output_dir='outputImage',
    file_name = 'result',
    export_format = 'png'
)
