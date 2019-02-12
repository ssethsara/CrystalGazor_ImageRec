import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def GetPhotoDataById(Userid):
    usersData = pd.read_csv('FacebookData/CGFC-Users.csv',)
    photoData = pd.read_csv('FacebookData/CGFC-Photos.csv',encoding='cp1252')

   
    usersData=usersData.loc[photoData['UserId'] == Userid]
    photoData=photoData.loc[photoData['UserId'] == Userid]

       
    photoData = photoData.drop('PhotoURL', 1)
    photoData = photoData.drop('Likes', 1)
    photoData = photoData.drop('Angry', 1)
    photoData = photoData.drop('Haha', 1)
    photoData = photoData.drop('Love', 1)
    photoData = photoData.drop('Sad', 1)
    photoData = photoData.drop('Wow', 1)
    photoData = photoData.drop('PostedBy', 1)
    
    
    return usersData,photoData  



