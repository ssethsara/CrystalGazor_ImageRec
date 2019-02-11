import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder






def CreateDummies(userData):
    dummies = pd.get_dummies(userData)
    #dummies.head()

    #Group the Dummies by the Photo ID
    grp_obj_df = dummies.groupby('Photo_ID').sum()
    return grp_obj_df,dummies.columns.values


   

def DataPreprocessing(userData):
      #Remove Unwanted Columns
  #obj_df = userImageData.select_dtypes(include=['object']).copy()
  if('Updated_Date' in userData.columns.values):
    print('Removed unwanted column')
    del userData["Updated_Date"]
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

def addToTable(data):
      
    #Empty dataframe to store filtered data
    aprioritDf_itemsets = data
    
    aprioritDf_itemsets['Num_of_Items'] = aprioritDf_itemsets['itemsets'].apply(lambda x: len(x))
    
    #convert dataframe column value to int
    aprioritDf_itemsets.Num_of_Items = aprioritDf_itemsets.Num_of_Items.astype(int)    
    
    return aprioritDf_itemsets


def GenerateAprioriTable(grp_obj_df,min_support):
  #@title Default title text
  #Apply Apriori Algorithm
  from mlxtend.frequent_patterns import apriori
  aprioriApplied = apriori(grp_obj_df, min_support, use_colnames = True)
  #print(aprioriApplied)
  #Convert panda series dataset to panda Dataframe
  aprioriApplied_df = pd.DataFrame({'support':aprioriApplied.support, 'itemsets':aprioriApplied.itemsets})
  aprioriApplied_Table = addToTable(aprioriApplied_df);
  return aprioriApplied_Table
  #aprioriApplied_Table.to_csv('aprioriApplied_Table.csv')


#To Filter by Number of Items
def FilterByNumOfItems(data,minSupport,numOfItems):
  #Get the No of Items
  filtered_By_Num_of_Items = data.loc[(data['Num_of_Items'] == numOfItems)]
  #Sort Data in Decending Order 
  filtered_By_Num_of_Items = filtered_By_Num_of_Items.sort_values(by = ['support',],ascending = False) 
  if(minSupport!=0):
    filtered_By_Num_of_Items = filtered_By_Num_of_Items.loc[(filtered_By_Num_of_Items['support'] >= minSupport)]
  return filtered_By_Num_of_Items 

def GetItemSetOnNumbers(number,aprioriApplied_Table):
  minSupport = 0
  #Filter data by Num_of_Items = 1
  return FilterByNumOfItems(aprioriApplied_Table,minSupport,number)

def PruningAlgo(aprioriApplied_Table):
  x=1
  while x <= (aprioriApplied_Table['Num_of_Items'].max()):
    prouningItems=GetItemSetOnNumbers(x,aprioriApplied_Table)
    previous=prouningItems.copy()
    if prouningItems.empty:
      prouningItems=previous
      return prouningItems
    x=x+1
  return prouningItems
   


def findAssociations(ItemsAttributes,final_result):
  AssociationList=[]
  for index, row in final_result.iterrows():
    if len([i for i, v in enumerate(row[0]) if v in ItemsAttributes])==len(ItemsAttributes):
      AssociationList.append({'itemset':[list(x) for x in enumerate(row[0])],'support':row['support']})
  return  AssociationList 


"""
def expandValues():
    itemNo = 0
    expandedData = pd.DataFrame()
    for index, row in filtered_By_Num_of_Items_3.iterrows():
        itemNo = itemNo+1
        items = list(row['itemsets'])
        for item in items:
            expandedValue = item.split('_')
            expandedData = expandedData.append({'itemNo':itemNo,'id':index,'Part' : expandedValue[0],'attribute' : expandedValue[1],'value' : expandedValue[2], 'support' : row['support'],'Num_of_Items' : row['Num_of_Items'] }, ignore_index = True)

        
    #convert dataframe column value to int
    expandedData.numberOfitems = expandedData.Num_of_Items.astype(int) 
    expandedData.id = expandedData.id.astype(int) 
    expandedData.itemNo = expandedData.itemNo.astype(int) 

    return expandedData
"""



def Apriori_Cal_Run(userData):
    #userData = pd.read_csv('ProcessedData.csv')
    userData=userData.loc[:, ~userData.columns.str.contains('^Unnamed')]
    userData=DataPreprocessing(userData)
    grp_obj_df,dummyColumn=CreateDummies(userData)  
    aprioriApplied_Table=GenerateAprioriTable(grp_obj_df,0.3)

    filtered_By_Num_of_Items_1=GetItemSetOnNumbers(1,aprioriApplied_Table)
    filtered_By_Num_of_Items_2=GetItemSetOnNumbers(2,aprioriApplied_Table)
    filtered_By_Num_of_Items_3=GetItemSetOnNumbers(3,aprioriApplied_Table)
    filtered_By_Num_of_Items_4=GetItemSetOnNumbers(4,aprioriApplied_Table)
    filtered_By_Num_of_Items_5=GetItemSetOnNumbers(5,aprioriApplied_Table)
    filtered_By_Num_of_Items_6=GetItemSetOnNumbers(6,aprioriApplied_Table)

    filtered_Item_Collection=PruningAlgo(aprioriApplied_Table)  
    result = [filtered_By_Num_of_Items_2, filtered_By_Num_of_Items_3,filtered_By_Num_of_Items_4,filtered_By_Num_of_Items_5]
    
    final_result = pd.concat(result)
    
    

    #print(AssociationList[0][0]['itemset'][0][1])
    """
    print('')
    print('######################################')
    print('')
    HighValue=PruningAlgo(aprioriApplied_Table)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(HighValue)
    print('')
    print('######################################')
    print('')
    """
    return final_result,dummyColumn


def CheckAssociationRules(ItemsAttributes,final_result): 
    AssociationList=findAssociations(ItemsAttributes,final_result)  
    return AssociationList    

#userData = pd.read_csv('ProcessedData2.csv')
#Apriori_Cal_Run(userData)