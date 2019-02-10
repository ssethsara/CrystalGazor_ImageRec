######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
from os.path import join

#color detection 1
#import__color recognition
from sklearn.cluster import KMeans
from sklearn import metrics


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from CGFC_functions import colorDetector as color_Detector
from CGFC_functions import category_Dic 
from CGFC_functions import CGFCConfig



# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'




# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 24

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value



def cropDetectedCloths(image,bbox):
    #Crop image by bbox
    ymin = bbox[0]
    xmin = bbox[1]
    ymax = bbox[2]
    xmax = bbox[3]

 

    (im_height,im_width,im_color) = image.shape
    (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)

    bboxWidth=xmaxx-xminn
    bboxHeight=ymaxx-yminn
    crop_img = image[int(yminn+(bboxHeight*2/10)):int(ymaxx-(bboxHeight*2/10)), int(xminn+(bboxWidth*2/10)):int(xmaxx-(bboxWidth*2/10))]
    cv2.imshow("cropped", crop_img)

    return crop_img



def Detect_Cloths(image):
    # image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})


    output = pd.DataFrame(
    {'image' : [image],
    'boxes' : [boxes],
    'classes' : [classes],
    'scores' : [scores] })

    #print(output['scores'][0])
    
    return output 
  

def colorRecognition(image,bbox):
    #color-Recognition 1
        dominet_colors=color_Detector.dominant_color_detector(crop_img,3)



def ClothDetectionAnalyse(image,tagData,gender):
    min_score_thresh=CGFCConfig.min_score_thresh
    detectedData=Detect_Cloths(image)

    boxes=detectedData['boxes'][0]
    scores=detectedData['scores'][0]
    classes=detectedData['classes'][0]


    bestResults=[]
    bestBBox=[]
    bestScores=[]
    bestClasses=[]
    UpperOrLower=[]
   
    normBBoxes=np.squeeze(boxes)
    normScores=np.squeeze(scores)
    normClasses=np.squeeze(classes)

    isLowerBodyClothAdded=False
    isUpperBodyClothAdded=False
    for index,className in enumerate(normClasses):
        className=category_index[className]['name']
       #if score>=min_score_thresh:
        if((gender=='Male') & (className not in category_Dic.Female_Cloths)&(className not in category_Dic.Attributes)):
                
                if((className in category_Dic.UpperBody) & (isUpperBodyClothAdded==False)):
                    UpperOrLower.append("Upperbody")
                    bestResults.append(index)
                    bestBBox.append(normBBoxes[index])
                    bestScores.append(normScores[index])
                    bestClasses.append(normClasses[index])
                    print("isUpper male:",className)
                    isUpperBodyClothAdded=True;
                elif((className in category_Dic.LowerBody) & (isLowerBodyClothAdded==False)):
                    UpperOrLower.append("LowerBody")
                    bestResults.append(index)
                    bestBBox.append(normBBoxes[index])
                    bestScores.append(normScores[index])
                    bestClasses.append(normClasses[index])
                    isLowerBodyClothAdded=True;
                    print("isLower male:",className)
                if((isLowerBodyClothAdded==True) & (isUpperBodyClothAdded==True)):
                    break
        elif((gender=='Female') & (className not in category_Dic.Attributes)): 
                if((className in category_Dic.UpperBody) & (isUpperBodyClothAdded==False)):
                    UpperOrLower.append("Upperbody")
                    bestResults.append(index)
                    bestBBox.append(normBBoxes[index])
                    bestScores.append(normScores[index])
                    bestClasses.append(normClasses[index])
                    print("isUpper Female :",className)
                    isUpperBodyClothAdded=True;
                elif((className in category_Dic.LowerBody) & (isLowerBodyClothAdded==False)):
                    UpperOrLower.append("LowerBody")
                    bestResults.append(index)
                    bestBBox.append(normBBoxes[index])
                    bestScores.append(normScores[index])
                    bestClasses.append(normClasses[index])
                    isLowerBodyClothAdded=True;
                    print("isLower Female:",className)
                if((isLowerBodyClothAdded==True) & (isUpperBodyClothAdded==True)):
                    break
    className=category_index[normClasses[index]]['name']
    print(className)                

    for index,score in enumerate(normScores):
      
       if ((score>=min_score_thresh) &(className in category_Dic.Attributes)):
            bestResults.append(index)
            bestBBox.append(normBBoxes[index])
            bestScores.append(score)
            bestClasses.append(normClasses[index])
         
    

  
    crop_image_Data = pd.DataFrame()
    
    for index,bbox in enumerate(bestBBox):
        
        crop_img=cropDetectedCloths(image,bbox)
        dominet_colors=color_Detector.dominant_color_detector(crop_img,3)
        colors=[]
        colorMax=dominet_colors[0]
        #print("dominet_colors : ",dominet_colors)
        for color in dominet_colors:
            #get Only one value
          
            if(color[1]>colorMax[1]):
    
                colorMax=color
            
                
        className=category_index[bestClasses[index]]['name']
        clothType=None
        clothStyle=None

  
        if (className in category_Dic.Attributes):
            clothType=className
            clothStyle=None
        else:     
            clothType,clothStyle=className.split("_")

        print("Final color : ",colorMax)
        
        uploadedDate=str(tagData["UploadedDate"])
        photoID=tagData["PhotoID"]
        
        crop_image_Data=crop_image_Data.append(pd.DataFrame(
            {'image' : tagData['PhotoName'],
             'type' : clothType,
             'Upper/Lower' : UpperOrLower[index],
             'style' : clothStyle,
             'scores' : bestScores[index],
             'dominant_colors': [color[3]],
             'UploadedDate':str(tagData["UploadedDate"]),
             'PhotoID':photoID}),ignore_index=True)
             
        
 
     

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
            detectedData['image'][0],
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=1,
            min_score_thresh=0.5)

    # All the results have been drawn on image. Now display the image.
    #cv2.imshow('Object detector', detectedData['image'][0])
    cv2.imwrite("FacebookData/Detected/Detected-"+str(tagData['PhotoID'])+".jpg", detectedData['image'][0])
  
    return crop_image_Data


    
