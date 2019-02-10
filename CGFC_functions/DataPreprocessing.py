import pandas as pd
import numpy as np

def DataPreprocessing(userData,min_thresh):
    userData=userData.loc[:, ~userData.columns.str.contains('^Unnamed')]
    #userData.drop(userData.columns[0], axis=1, inplace=True)
    upperBodyOnly=userData.loc[userData['Upper/Lower']=='Upperbody'].copy()
    lowerBodyOnly=userData.loc[userData['Upper/Lower']=='LowerBody'].copy()

    upperBodyOnly=upperBodyOnly.reset_index(drop=True)
    lowerBodyOnly=lowerBodyOnly.reset_index(drop=True)
    lower=lowerBodyOnly.copy()
    upper=upperBodyOnly.copy()

    
    c = pd.DataFrame()
    newData=pd.DataFrame()
    for index, row in lowerBodyOnly.iterrows():
        photoID=row['PhotoID']
        updatedDate=row['UploadedDate']
        
        if upper.iloc[index]['scores']<min_thresh:
            upper.iloc[index]='-'
        if (lower.iloc[index]['scores'] < min_thresh):
            lower.iloc[index]='-'
            
            
        newData=newData.append(pd.DataFrame({
            "Photo_ID":photoID,
            "Updated_Date":updatedDate,
            "Upperbody_Type":[upper.iloc[index]['type']],
            "Upperbody_Color":[upper.iloc[index]['dominant_colors']],
            "Upperbody_Style":[upper.iloc[index]['style']],
            "Lowerbody_Type":[lower.iloc[index]['type']],
            "Lowerbody_Color":[lower.iloc[index]['dominant_colors']],
            "Lowerbody_Style":[lower.iloc[index]['style']]
        })) 
        
        
    return newData

#userData=pd.read_csv('FBData_results.csv')
#data=DataPreprocessing(userData,0.5)
#print(data)