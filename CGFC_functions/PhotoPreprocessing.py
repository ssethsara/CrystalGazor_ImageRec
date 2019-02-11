#import sys
import pandas as pd
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

#sys.path.insert(0, '../')

#import FacebookData

def CropTaggedPerson(image,tagDf):

    height, width, channels = image.shape 
  
    
    #extract tag position data
    Tag_width=float(tagDf["Tag_width"])
    Tag_height=float(tagDf["Tag_height"])
    Tag_left=float(tagDf["Tag_left"])
    Tag_top=float(tagDf["Tag_top"])

    

    #convert to x,y axis
    heightPixel=(height*Tag_height/100)
    widthPixel=(width*Tag_width/100)

    #head position
    y1_head=int(height*Tag_top/100)
    x1_head=int(width*Tag_left/100)
    y2_head=int(y1_head+heightPixel)
    x2_head=int(x1_head+widthPixel)

    #body position
    y1_body=y1_head
    x1_body=int(x1_head-widthPixel*0.75)
    y2_body=int(y1_head+((height*Tag_height/100))*7.5)
    x2_body=int(x1_head+((width*Tag_width/100))*1.75)

    if(x1_body<0):
        x1_body=0
    if(y2_body>height):
        y2_body=height
    if(x2_body>width):
        x2_body=width
           

    cv2.rectangle(image, (x1_body, y1_body), (x2_body, y2_body), (0,255,0), 2)
    #cv2.rectangle(image, (x1_head, y1_head), (x2_head, y2_head), (255,0,0), 1)
    print("tagged")

    crop_img = image[int(y1_body):int(y2_body), int(x1_body):int(x2_body)]
    
    #cv2.imshow('image', crop_img)

    # Press any key to close the image
    #cv2.waitKey(0)

    return crop_img




    
    
