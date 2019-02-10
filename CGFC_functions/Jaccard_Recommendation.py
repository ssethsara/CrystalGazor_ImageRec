import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist


#https://github.com/seatgeek/fuzzywuzzy
from fuzzywuzzy import fuzz

import webbrowser



def DataPreprocessing(userData):
      #Remove Unwanted Columns
  #obj_df = userImageData.select_dtypes(include=['object']).copy()
  if('UploadedDate' in userData.columns.values):
    print('Removed unwanted column')
    del userData["UploadedDate"]
  if('image' in userData.columns.values):
    print('Removed unwanted column')
    del userData["image"]  
  if('User_ID' in userData.columns.values):
    print('Removed unwanted column')
    del userData["User_ID"] 
  if('Image_Link' in userData.columns.values):
    print('Removed unwanted column')
    del userData["Image_Link"]
  if('Image_Rating' in userData.columns.values):
    print('Removed unwanted column')
    del userData["Image_Rating"] 
    
  #Replace Null Values with NaN
  userData=userData.replace(['-'],[np.nan])

  #null Remove make it itterative
  for attribute in userData.columns.values:
    userData[attribute].value_counts()
    macValueIndex=userData[attribute].value_counts().idxmax()
    userData = userData.fillna({attribute: macValueIndex})
    userData[userData.isnull().any(axis=1)]
    
  return userData

def CreateDummies(userData):
  #Convert All the Attributes into Binary
  dummies = pd.get_dummies(userData)
  #Concat the Attributes with Dummies
  userData['g']=userData.groupby('Photo_ID').cumcount()
  dummies['g']=dummies.groupby('Photo_ID').cumcount()
  obj_df_with_dummies=userData.merge(dummies,how='outer').drop('g',1)
  return obj_df_with_dummies
  
    


def CalculateJaccardSimilarity(dummygrp):
  #Jaccard Distance
  dist = pdist(dummygrp, metric="jaccard")
  s_dist = squareform(dist)
  # Fill diagonals with nulls
  np.fill_diagonal(s_dist, 1)
  #Jaccard Similarity
  sim = np.subtract(1, s_dist)
  sim_df = pd.DataFrame(sim, columns=dummygrp.index, index=dummygrp.index)
  return sim_df
 
def GetSimilaritySummation(sim_df,userData):
    #Sum of the Jaccard Indexes under each Column
    sum_colum=sim_df[1].sum()
    sum_colum
    sum_colum_totals=[]
    photo_ids=[]
    for photo_id in sim_df:
        sum_colum=sim_df[photo_id].sum()
        sum_colum_totals.append(sum_colum)
        photo_ids.append(photo_id)

    list={"Photo_ID":photo_ids,"Similarity":sum_colum_totals}  
    sim_df_with_total = pd.DataFrame(list,index=photo_ids) 

    #Merge Sums with the Correspondent Record - under the ID
    userData['g']=userData.groupby('Photo_ID').cumcount()
    sim_df_with_total['g']=sim_df_with_total.groupby('Photo_ID').cumcount()
    sum_colum_totals_withID=userData.merge(sim_df_with_total,how='outer').drop('g',1)

    sum_colum_totals_withID
    return sum_colum_totals_withID

def FilterValues(sim_df_with_total):
    #Filter out Max n Records
    filterMax = sim_df_with_total.nlargest(10, 'Similarity')
    filterMax
    return filterMax

def getMostPreferedCloths_LowerBody(filterMax):
    lowerBody_df = filterMax.filter(items=['Photo_ID','Lowerbody_Type', 'Lowerbody_Color', 'Lowerbody_Style','Similarity'])
    lowerBody_df=lowerBody_df.drop_duplicates()
    lowerBody_df=lowerBody_df.reset_index(drop=True)
    lowerBody_df=lowerBody_df.loc[0]

    lowerBodyPref_Color=lowerBody_df['Lowerbody_Color']
    lowerBodyPreference= str(lowerBody_df['Lowerbody_Type']+'_'+lowerBody_df['Lowerbody_Style'])
    lowerBodyResult=[lowerBodyPreference,lowerBodyPref_Color]
    return lowerBodyResult


def getMostPreferedCloths_UpperBody(filterMax):
    upperBody_df = filterMax.filter(items=['Photo_ID','Upperbody_Type', 'Upperbody_Color', 'Upperbody_Style','Similarity'])
    upperBody_df=upperBody_df.drop_duplicates()
    upperBody_df=upperBody_df.reset_index(drop=True)
    upperBody_df=upperBody_df.loc[0]

    upperBodyPref_Color=upperBody_df['Upperbody_Color']
    upperBodyPreference= str(upperBody_df['Upperbody_Type']+'_'+upperBody_df['Upperbody_Style'])
    upperBodyResult=[upperBodyPreference,upperBodyPref_Color]
    return upperBodyResult

    

def PreProcessStoreDataset(store):
    storeColumns=list(store.columns.values)
    StorerowString=pd.DataFrame()
    for index, row in store.iterrows():
        #rowString=(row['Item_Name'])+' '+' '.join(str(e) for e in row['Details'].split(','))
        rowString=' '.join(str(e) for e in row['Details'].split(','))
        StorerowString=StorerowString.append(pd.DataFrame({
            'ItemID':index,
            'SearchString':rowString,
            'Color':[str(row['Color'])]
        }),ignore_index=True)
        #StorerowString  (parameter_list):
    return StorerowString


def ListDownSimilarWords(mpcParam,similarWords):
  cloth=[]
  for cloth in mpcParam:
    similarClothName=[]
    del similarClothName[:]
    clothName=(cloth[0].replace('_'," "))
    print(clothName)
    if(cloth[0] in similarWords.columns.values):
      similarClothNames=(list(similarWords[cloth[0]][similarWords[cloth[0]].notnull()]))+[clothName]
      #similarClothNames=similarClothNames
    else:
      similarClothNames=similarClothName+[clothName]
      
    cloth=cloth.append(similarClothNames)   
  return mpcParam 

def getSearchScore(rowString,clothItem):
    #rowString=StorerowString.loc[0]['SearchString']
    #print(rowString)
    fuzz_Scores=[]
    for simWords in clothItem[2]:
        score=fuzz.token_set_ratio(simWords.lower(), rowString.lower())
        fuzz_Scores.append(int(score))
    #print(fuzz_Scores)   

    return max(fuzz_Scores)

    #maxScore=getSearchScore(rowString,mpc)
    #print(maxScore)  


def ScoreShopItems(StorerowString,cloths):
    ShopDatasetUpdated=pd.DataFrame()
    for index,item in StorerowString.iterrows():
        maxScore=getSearchScore(item['SearchString'],cloths)
        #item['fuzz_score']=maxScore
        #print(maxScore,'-',item['SearchString'])
        StorerowString['maxScore']=maxScore
        ShopDatasetUpdated=ShopDatasetUpdated.append(pd.DataFrame({
            'ItemID':[item['ItemID']],
            'Score':[maxScore]
        }))

    return ShopDatasetUpdated  



def JaccardRecommendationRun(userData):

    pd.options.display.max_colwidth = 100
    #userData = pd.read_csv('Sid_Original.csv')
    #SimilarWords for Cloth Types
    similarWords=pd.read_csv('CGFC_functions\similar-words.csv')

    #Store database
    store = pd.read_csv('CGFC_functions\OnlineStore.csv')
    store = store.replace('\n',',', regex=True)
    store.head()

    userData=userData.loc[:, ~userData.columns.str.contains('^Unnamed')]
    userData=DataPreprocessing(userData)
    dummyDf=CreateDummies(userData) 
    dummygrp = dummyDf.groupby('Photo_ID').sum()
    sim_df=CalculateJaccardSimilarity(dummygrp)
    sim_df_with_total=GetSimilaritySummation(sim_df,userData) 
    filterMax=FilterValues(sim_df_with_total)   
    lowerBodyResult=getMostPreferedCloths_LowerBody(filterMax)
    print(lowerBodyResult) 
    upperBodyResult=getMostPreferedCloths_UpperBody(filterMax)
    print(upperBodyResult)

    MostPreferedCloths=[upperBodyResult.copy(),lowerBodyResult.copy()]
    StorerowString=PreProcessStoreDataset(store)

    mpc=pd.DataFrame()
    mpc=MostPreferedCloths.copy() 

    mpcList=ListDownSimilarWords(mpc,similarWords)   
      

    for cloths in mpc:
        ShopDatasetUpdated=ScoreShopItems(StorerowString,cloths)  

        MaxScoreItems=ShopDatasetUpdated.nlargest(5,'Score')
        itemList=list(MaxScoreItems['ItemID'])
       
        RecommendedShopItems=store.iloc[itemList]
        #RecommendedShopItems=RecommendedShopItems.loc[RecommendedShopItems['Color'].str.lower()==mpc[0][1].lower()]
      
        print(RecommendedShopItems['URL'].iloc[:5].values) 
        webbrowser.open_new_tab(str(RecommendedShopItems['URL'].iloc[:5].values[0]))
        

        
#userData=pd.read_csv('Sid_Original.csv')
#JaccardRecommendationRun(userData)     
