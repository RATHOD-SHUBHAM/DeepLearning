# Social distancing Monitoring System:
python version: v3.7.7

# Installation and Setup procedure:
   1. Download the project zip file and extract it
   2. Open cmd and navigate to the root directory(social-distancing-detector) 
   3. Now we need to download the pretrained weights file using the link mentioned below. 
      We have used YOLOv3-320 weights. This can be downloaded from the link: https://pjreddie.com/darknet/yolo/
      links for the weight files are as follows:
      YOLOv3-320: https://pjreddie.com/media/files/yolov3.weights
      once the file is downloaded, place it under the directory, /model_dataset_files
   4. Install the dependencies using the following command:
      pip install -r requirements.txt
   5. Execute the main social distancing detector file using the following command:
      python social_distancing_detector.py --input input.mp4 --output output.mp4 --display 0
      (set display to 1 if you want to see output video as processing occurs)

# Command parameter explaination:
   python python_file --input input_file --output output_file --display display_flag
parameters:
   input_file     (optional) : path to input video file
   output_file    (optional) : path to output video file
   display_flag   (optional) : flag to determine if the output frame window needs to be displayed (default=1)

# Project structure:
   1. /model_dataset_files (selecting right weights: tradeoff between speed and accuracy)
      this directory contains the YOLO pretrained model(weights and the coniguration files)
      and also the file which contains the Coco class labels
   2. detection.py
      file contains function to detect people, draw a bounding box around them, define centroid and return results
   3. requirements.txt
      file contains dependencies required for the code to run
   4. social_distancing_detector.py
      the main python file which contain the code, that defines the project

# Project feature breakdown:
   1. Object detection using the YOLO COCO model to detect only people in a video stream.
   2. Computes the pairwise distances between all detected people.
   3. Based on the computed distances, we determine whether social distancing rule is being violated or not.
