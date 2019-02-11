import os
from os.path import join
import cv2
import pandas as pd
import numpy as np



from FacebookData import FacebookData as fbData
from CGFC_functions import PhotoPreprocessing as photo_preprocess
import Object_detection_image as oDI
from CGFC_functions import CGFCConfig
from CGFC_functions import DataPreprocessing
from CGFC_functions import Jaccard_Recommendation
from CGFC_functions import Apriori_Cal
from CGFC_functions import PhotoRating

from sklearn.externals import joblib

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

    return ExtractedData    



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
      
      
def CGFC_Start():
    print('')
    print('Rating on Reactions and Comments')
    print('******************************************************************************************')
    #df = pd.read_csv("Fb_reaction.csv", encoding='cp1252')
    userReactions=pd.read_csv('CGFC_functions\FB_reaction.csv')
    reactRating=PhotoRating.RatingByReactions(userReactions)
    userComments = pd.read_csv("CGFC_functions\Fb_comments.csv", encoding='utf-8')
    commentRating=PhotoRating.RatingByComments(userComments)
    rating=pd.merge(reactRating, commentRating, on="PhotoID")
    print(rating)  
    print('') 
    print('')
    print('******************************************************************************************')
    print('')
    min_thresh=0.3
    #------------saved File ------------
    userData=pd.read_csv('FacebookData\Supun Sethsara_FBData_results.csv')
    #-----------Detection----------------    
    #userData=analysePhotoCollection(1)
    data=DataPreprocessing.DataPreprocessing(userData,min_thresh)
    print('')
    print('Image classification Results')
    print('******************************************************************************************')
    print('')
    print(data)
    print('')
    print('******************************************************************************************')
    print('')
    print('')
    print('Recomondation ')
    print('******************************************************************************************')
    print('')
    reccom=Jaccard_Recommendation.JaccardRecommendationRun(data) 
    print(reccom) 
    print('')
    print('******************************************************************************************')
    print('')
    print('Association Rules ')
    print('******************************************************************************************')
    print('')
    final_result,fRcolumns=Apriori_Cal.Apriori_Cal_Run(data)     
    print('')
    print('******************************************************************************************')
    print('')
    print('Check Associations')
    print('******************************************************************************************')
    print('')
    ItemsAttributes=['Lowerbody_Type_trouser','Lowerbody_Style_Denim'] 
    AssociationList=Apriori_Cal.CheckAssociationRules(ItemsAttributes,final_result)   
    for associ in AssociationList:
        print(associ['itemset'])  
    print('')
    print('******************************************************************************************')
    print('')

CGFC_Start()