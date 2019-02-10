import os
from os.path import join
import cv2
import pandas as pd
import numpy as np



from FacebookData import FacebookData as fbData
from CGFC_functions import PhotoPreprocessing as photo_preprocess
import Object_detection_image as oDI
from CGFC_functions import CGFCConfig

def analysePhotoCollection(userId):
    ExtractedData=pd.DataFrame()

    usersData,photoData=fbData.GetPhotoDataById(userId)
    print(len(photoData))

    for index, selectedPhotoData in photoData.iterrows():
        tagData=selectedPhotoData[['PhotoID','PhotoName','UploadedDate','Tag_width','Tag_height','Tag_left','Tag_top']]
        userName=usersData['Name'][0]
        gender=usersData['Gender'][0]
        imageName=selectedPhotoData['PhotoName']
        photoID=selectedPhotoData['PhotoID']

        print("PhotoID :",photoID)
        print("Username :",userName)
        print("imageName :",imageName)
    
        imagePath='FacebookData/'+userName+'/photos/'
        image = cv2.imread(join(imagePath,imageName))


        tagged_image=photo_preprocess.CropTaggedPerson(image,tagData)


        #tagged_image = cv2.resize(tagged_image, (0,0), fx=0.5, fy=0.5)
        #image = cv2.imread(PATH_TO_IMAGE)

        onePhotoData=oDI.ClothDetectionAnalyse(tagged_image,tagData,gender)

        ExtractedData=ExtractedData.append(onePhotoData)
    

    ExtractedData.reset_index(inplace = True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(ExtractedData)

    fildir='FacebookData/'+userName+'_FBData_results.csv'

    if os.path.isfile(fildir):
            try:
                os.remove(fildir)
                # save new pic after this 
                ExtractedData.to_csv(fildir)
                print(fildir+' file Replaced')
            except:
                print('error: '+fildir+' File replace failed')
    else:
        ExtractedData.to_csv(fildir)



def AnalyseOnePhoto(photoNumber,userId):

    usersData,photoData=fbData.GetPhotoDataById(userId)

    selectedPhotoData=photoData.iloc[photoNumber]
    print(selectedPhotoData)
    tagData=selectedPhotoData[['PhotoID','PhotoName','UploadedDate','Tag_width','Tag_height','Tag_left','Tag_top']]
    userName=usersData['Name'][0]
    gender=usersData['Gender'][0]
    imageName=selectedPhotoData['PhotoName']
    photoID=selectedPhotoData['PhotoID']

    print("PhotoID :",photoID)
    print("Username :",userName)
    print("imageName :",imageName)
    
    imagePath='FacebookData/'+userName+'/photos/'
    image = cv2.imread(join(imagePath,imageName))


    tagged_image=photo_preprocess.CropTaggedPerson(image,tagData)

    onePhotoData=oDI.ClothDetectionAnalyse(tagged_image,tagData,gender)
    
    print(onePhotoData)
      
      


AnalyseOnePhoto(1,1)
#analysePhotoCollection(1)