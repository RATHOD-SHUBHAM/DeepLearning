# Project Name: Social Distancing Monitoring System
# Team number: 14
# python version: v3.7.7
# Team Members:
    # Abhishek Jain         (1001759977)
    # Nitish Prabhu Kota    (1001738851)
    # Shubham Shankar       (1001761068)



# importing the necessary libraries
import os
import numpy as np
import imutils
import argparse
import cv2 as openCv2
from detection import detect_people
from scipy.spatial import distance as dist

# file path definitions
# Loading the COCO class labels on which the YOLO model was trained
# derive the paths to the YOLO weights and model configuration
classFilePath = "model_dataset_files\coco.names"
modelWeightsPath = "model_dataset_files\yolov3-320.weights"
modelConfigPath = "model_dataset_files\yolov3.cfg"

# define the minimum safe distance (in pixels) that two people can be from each other
MIN_DISTANCE_IN_PIXELS = 450

# Argument parser construction and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="")
ap.add_argument("-o", "--output", type=str, default="")
ap.add_argument("-d", "--display", type=int, default=1)
args = vars(ap.parse_args())
# print(args)

classNames = open(classFilePath).read().strip().split("\n")

# loading the pretrained YOLO object detector which was trained on COCO dataset
# https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#gafde362956af949cce087f3f25c6aff0d
# Reads a network model stored in Darknet model files.
# Returns a Network object that ready to do forward
# creating a network using model config and weights
net = openCv2.dnn.readNetFromDarknet(modelConfigPath, modelWeightsPath)

# https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#a7f767df11386d39374db49cd8df8f59e
# Declaring OpenCV as backend and also declaring that we are using CPU
net.setPreferableBackend(openCv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(openCv2.dnn.DNN_TARGET_CPU)

# from YOLO, determining only the "output" layer names which are needed 
layerNames = net.getLayerNames()
print(layerNames)
# index starts from 1 and not 0 for layerNames , so we use -1 below
# output names of the layers
outputlayerNames = [layerNames[i[0] - 1]
                    for i in net.getUnconnectedOutLayers()]
print(outputlayerNames)


# Initialize a video stream and the output video file pointer
print("[INFO] video stream access started...")
# open input video if available else webcam stream
# OpenCV uses webcam 0 by default
# openCv2.VideoCapture() - index=0 for webcam, can give index = 1 or other if we have multiple cameras. or can give a video file name as well
cap = openCv2.VideoCapture(args["input"] if args["input"] else 0)
output_writer_variable = None

# Loop from the video stream, frame by frame
# create a while loop to get frames of our video or webcam, based on the scenario
while True:
    # read the next frame from the input video
    (success, frame) = cap.read()

    # openCv2.imshow('Image', frame)
    # openCv2.waitKey(10000)

    # if the frame was not captured(if the camera is not available or if the input file is not provided), then it breaks out of the loop
    if not success:
        break

    # method to resize the frame
    frame = imutils.resize(frame, width=720)
    # method to detect object(people in our case), this returns a result array 
    # this array contains confidence level, bounding box and centroid values
    detect_people_result_variable = detect_people(frame, net, outputlayerNames,
                            classIdx=classNames.index("person"))

    # The set of indexes that violate the minimum social distance are initialized
    violation_set = set()

    # we need atleast 2 people detections to compute pairwise distance maps
    if len(detect_people_result_variable) >= 2:
        # from the results, we extract all centroid values to find eucledean distances between them(pairwise)
        value_of_centroid = np.array([r[2] for r in detect_people_result_variable])
        D = dist.cdist(value_of_centroid, value_of_centroid, metric="euclidean")

        # looping through the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                # condition to check if 2 objects are not within the given pixel value_of_centroid
                if D[i, j] < MIN_DISTANCE_IN_PIXELS:
                    # centroid pair index is added to the violation set
                    violation_set.add(i)
                    violation_set.add(j)

    # result array is looped over
    for (i, (prob, bbox, centroid)) in enumerate(detect_people_result_variable):
        # store the coordinates of centroid and bounding box in variables
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        # color is set to Green
        color_object = (0, 255, 0)

        # update the color if the pair value exists within the violation set
        if i in violation_set:
        # color is set to Red
            color_object = (0, 0, 255)

        # this method draws a rectangle around the person detected with the provided color 
        openCv2.rectangle(frame, (startX, startY), (endX, endY), color_object, 2)
        # this method draws a circle within the rectangle drawn, with the provided color
        openCv2.circle(frame, (cX, cY), 5, color_object, 1)

    # Draw on the output frame the cumulative number of social distancing violations
    violationTextOnFrame = "Violation count: {}".format(len(violation_set))
    openCv2.putText(frame, violationTextOnFrame, (10, frame.shape[0] - 25),
                openCv2.FONT_ITALIC, 0.8, (0, 0, 255), 3)

    # If condition to check if the output frame should be shown on the screen or not
    # output frame is shown if its true. else dont show the frame
    if args["display"] > 0:
        openCv2.imshow("Social Distancing Monitoring System Output", frame)
        variable_key_value = openCv2.waitKey(1) & 0xFF

        # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
        # break from the loop, if the key 'x' is pressed
        if variable_key_value == ord("x"):
            break

    # check to see if the output path is passed and the writer is initialized
    # if not initialized yet, then we initialize the writer variable
    if args["output"] != "" and output_writer_variable is None:
        # video writer is initialized if its not done yet
        # https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
        variable_writer = openCv2.VideoWriter_fourcc(*"MJPG")
        output_writer_variable = openCv2.VideoWriter(args["output"], variable_writer, 25, (frame.shape[1], frame.shape[0]), True)

    # If there is no video writer, write a frame to the output video file.
    if output_writer_variable is not None:
        print("[INFO] video stream has been processed and is written to the output")
        output_writer_variable.write(frame)
