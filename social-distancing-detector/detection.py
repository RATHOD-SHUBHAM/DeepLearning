# imports of numpy and OpenCV
import numpy as np
import cv2 as openCv2

# To root out false detections or bad ones, initialize the minimum confidence value
valueOfminimumConfidence = 0.3
# Defining a threshold variable to be used when implementing non-maxim suppression
thresholdValueNMS = 0.3
#variable to store Width Height of the Target output frame
widthHeightVariable=320

# this is a function which detects people,
# passes the result which contains confidence value, bounding box and the centroid values in an array
def detect_people(imgFrame, model_net_value, outputlayerNames, classIdx=0):
    # grab dimensions of the frame and initialize the list of results
    (H, W) = imgFrame.shape[:2]
    detect_people_result_variable = []

    # Create a blob from the input frame and then execute a forward pass of the YOLO object detector,
    # which shows the Related likelihoods and the bounding boxes
    # network accepts only blob formats. so we convert it first
    # https://docs.opencv.org/master/d6/d0f/group__dnn.html
    # Creates 4-dimensional blob from image.
    # (input frame, scaling factor, target size)
    object_blob = openCv2.dnn.blobFromImage(imgFrame, 1/255, (widthHeightVariable, widthHeightVariable), swapRB=True, crop=False)
    # set blob as input to our network
    model_net_value.setInput(object_blob)
    # send the image as a forward pass to our network; and we can find output of the 3 layers ['yolo_82', 'yolo_94', 'yolo_106']
    # 3 outputs can be used to find 3 bounding boxes
    output_from_layer = model_net_value.forward(outputlayerNames)
    # print(output_from_layer[0].shape)
    # print(output_from_layer[1].shape)
    # print(output_from_layer[2].shape)
    # print(output_from_layer[0][0])

    # variable initialization for confidence, bounding boxes, centroids
    array_bounding_box = []
    array_centroids = []
    array_confidences = []

  
    for output_array_loop in output_from_layer:
        # loop over each of the detections
        for array_of_detection_loop in output_array_loop:
            # print(detection)
            # from the current object detected, we extract class ID and confidence of the detection
            variable_scores = array_of_detection_loop[5:]
            variable_classID = np.argmax(variable_scores)
            variable_confidence = variable_scores[variable_classID]

            
            # if condition to check if the confidence is greater than minimum value specified
            # and to check if the classID equals the person class
            if variable_classID == classIdx and variable_confidence > valueOfminimumConfidence:
                # bounding box coordinates scaling
                box_stored_variable = array_of_detection_loop[0:4] * np.array([W, H, W, H])
                (variable_centerX, variable_centerY, variable_width, variable_height) = box_stored_variable.astype("int")

                # To obtain the top and left corner of the bounding box, use the Center (X,Y) coordinates
                value_X = int(variable_centerX - (variable_width / 2))
                value_Y = int(variable_centerY - (variable_height / 2))

                # array values for the below arrays are updated
                array_bounding_box.append([value_X, value_Y, int(variable_width), int(variable_height)])
                array_centroids.append((variable_centerX, variable_centerY))
                array_confidences.append(float(variable_confidence))

    # NMS (non-maxima suppression) is applied
    # This is done to suppress weak, overlapping bounding boxes
    detection_count = openCv2.dnn.NMSBoxes(array_bounding_box, array_confidences, valueOfminimumConfidence, thresholdValueNMS)

    # check to see if atleast one detection has occured
    if len(detection_count) > 0:
        # detection indexes are looped
        for i in detection_count.flatten():
            # bounding box coordinates is obtained and stored
            (value_X, value_Y) = (array_bounding_box[i][0], array_bounding_box[i][1])
            (w, h) = (array_bounding_box[i][2], array_bounding_box[i][3])

            # results array is updated which consists to consists confidence values, bounding box coordinates and centroid
            result_Array = (array_confidences[i], (value_X, value_Y, value_X + w, value_Y + h), array_centroids[i])
            detect_people_result_variable.append(result_Array)

    # return the list of results
    return detect_people_result_variable
