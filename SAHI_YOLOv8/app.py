import streamlit as st
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
import os

# todo: Register the model
yolov8_model_path = "weights/yolov8x.pt"

# todo: Standard Inference with a YOLOv8 Model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.25,
    device="cpu", # or 'cuda:0'
)
# Save input image
def save_uploadedfile(uploadedfile):
    with open(os.path.join("inputImage", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to inputImage".format(uploadedfile.name))

image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
if image_file is not None:
    if st.button('RUN'):

        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        st.write(file_details)
        save_uploadedfile(image_file)

        input_path = 'inputImage/'+ image_file.name

        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Input Image")
            st.image(input_path)

        # todo: Inference without SAHI
        result = get_prediction(input_path, detection_model)

        result.export_visuals(export_dir="outputImage/")

        with col2:
            st.header("YoloV8 Result")
            st.image("outputImage/prediction_visual.png")

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

        with col3:
            st.header("SAHI Result")
            st.image("outputImage/result.png")
