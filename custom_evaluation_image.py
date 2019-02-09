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
from CGFC_functions import PhotoPreprocessing as photo_preprocess
from CGFC_functions import category_Dic
from CGFC_functions import evaluation
from evaluationData import getEvalData 



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
    crop_img = image[int(yminn):int(ymaxx), int(xminn):int(xmaxx)]
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



def EvaluateObjectDetection(image,data,min_tresh_values):

    trueClass=data["class"]
    trueXmin=data["xmin"]
    trueXmax=data["xmax"]
    trueYmin=data["ymin"]
    trueYmax=data["ymax"]
    
    originalBB={'x1' : trueXmin,
                 'x2' : trueXmax,
                 'y1' : trueYmin,
                 'y2' : trueYmax }

    detectedData=Detect_Cloths(image)

    boxes=detectedData['boxes'][0]
    scores=detectedData['scores'][0]
    classes=detectedData['classes'][0]
    result=pd.DataFrame()
    for min_score_thresh in min_tresh_values:
   
        normBBoxes=np.squeeze(boxes)
        normScores=np.squeeze(scores)
        normClasses=np.squeeze(classes)

        iou=0.0
        detected=False
        data['DetectedClass'] = 'UnDetected'
        className=category_index[normClasses[0]]['name']
        checkAttributes=True
        
        if(trueClass in category_Dic.Attributes):
            checkAttributes=True
        elif(className in category_Dic.Attributes):
            checkAttributes=False
        
        if ((normScores[0]>min_score_thresh) & checkAttributes):
                data['DetectedClass']=className
                data['Confidence'] = normScores[0]
                data['loc_Accuracy']=0

        for index,className in enumerate(normClasses):
            className=category_index[className]['name']
            if(normScores[index]<min_score_thresh):
                data['Detected'] = False
                data['loc_Accuracy']=0
                break
            
            if ((className==trueClass)):
                
                detected=True
                #print(className+':'+str(normScores[index]))
                bbox=normBBoxes[index]
                ymin = bbox[0]
                xmin = bbox[1]
                ymax = bbox[2]
                xmax = bbox[3]
                (im_height,im_width,im_color) = image.shape
                (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                cv2.rectangle(image, (int(trueXmin), int(trueYmin)), (int(trueXmax), int(trueYmax)), (0,255,0), 2)
        
                detectedBB={'x1' : xminn,
                             'x2' : xmaxx,
                             'y1' : yminn,
                             'y2' : ymaxx }
            
                iou=evaluation.get_iou(originalBB, detectedBB) 
                #print(iou)
                data['DetectedClass']=className
                data['Detected'] = True
                data['Confidence'] = normScores[index]
                data['loc_Accuracy'] = iou
                break
            
        data['min_tresh']=min_score_thresh
        result=result.append(data)   
            
       
     # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
            detectedData['image'][0],
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=1,
            min_score_thresh=min_score_thresh)

    # All the results have been drawn on image. Now display the image.
   # cv2.imshow('Object detector', detectedData['image'][0])
   

    return result
       

    


def main():
    completeEvaluationData=pd.DataFrame()
    
    min_tresh_values=[0.2,0.4,0.6,0.8]

    for className in ['Coat_Regular','Dress_Casual','Dress_Formal','Dress_Party','High-Heel','Jacket_Regular','Sandal','Shirt_LongSleeves','Shirt_Polo','Shirt_Regular','Shoe','Short_Regular','Skirt_Long','Skirt_Short','Suit_Regular','Tie','Top_Casual','Top_Formal','trouser_Denim','trouser_Regular','trouser_Slim','T-Shirt_LongSleeves','T-Shirt_Regular']:
    #for className in ['Skirt_Short']:   
        testDataSet=getEvalData.GetObjectByClass(className)
        evaluationData=pd.DataFrame()
        print("className : ",className," - evaluating..")
        for selectedPhotoData in testDataSet.iterrows():
            #selectedPhotoData=testDataSet.iloc[photoNumber]
            selectedPhotoData=selectedPhotoData[1]
            imageName=selectedPhotoData['filename']

            #print("imageName :",imageName)
            imagePath='evaluationData/'+className
            image = cv2.imread(join(imagePath,imageName))
            #tagged_image = cv2.resize(image,(0,0), fx=0.5, fy=0.5)
            result=EvaluateObjectDetection(image,selectedPhotoData,min_tresh_values)
            evaluationData=evaluationData.append(result)
            print("imageName : ",imageName," - complete")

        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
         #  print (evaluationData)
        evaluationData['Detected'] = evaluationData['Detected'].astype('bool')
        evaluationData.loc_Accuracy = evaluationData.loc_Accuracy.astype(float)
        evaluationData['loc_Accuracy'].fillna(0, inplace=True)
        
        
        evaluationData['Confidence'].fillna(0, inplace=True)
        evaluationData['DetectedClass'].fillna('UnDetected', inplace=True)
        print('')
        print(className," AP evaluation.#############################################") 
        print('')
        evaluation.evaluate(evaluationData)
        print('')
        print("######################################################################") 
        print('')
        completeEvaluationData=completeEvaluationData.append(evaluationData)

    
    
    fileName='evaluationData/Evaluation_results.csv'  
    if os.path.isfile(fileName):
        try:
            os.remove(os.path.join('evaluationData/', 'Evaluation_results.csv'))
            # save new pic after this 
            completeEvaluationData.to_csv(fileName)
            print(fileName+'file Replaced')
        except:
            print('error:'+fileName+'File replace failed')
    else:
        completeEvaluationData.to_csv(fileName) 
    print('')
    print("Complete AP evaluation.######################################") 
    evaluation.evaluate(completeEvaluationData)   
    
    # Press any key to close the image
    #cv2.waitKey(0)

    # Clean up
    #cv2.destroyAllWindows()

main()    
