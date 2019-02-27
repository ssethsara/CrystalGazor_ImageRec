#Interfacce Created using tkinter library

from tkinter import *
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from tkinter.ttk import *
from pandastable import Table, TableModel
import cv2

import pandas as pd
import numpy as np

from FacebookData import FacebookData as fbData
from CGFC_functions import CGFCConfig
from CGFC_functions import DataPreprocessing
from CGFC_functions import Jaccard_Recommendation
from CGFC_functions import Apriori_Cal
from CGFC_functions import PhotoRating
import CGFC as cgfc


min_thresh=0.3
columns=['None','Select']
ApriotResults=pd.DataFrame()
gender='male'
userId=1

PPdata=pd.DataFrame()
photoData=pd.DataFrame()
fBphotoData=pd.DataFrame()
usersData=pd.DataFrame()


#Get and store photo data and User data when start 
def initiate():
    global fBphotoData
    global usersData
    usersData,fBphotoData=fbData.GetPhotoDataById(userId)   
      
initiate()

#Set user By id
def SetUser():
     global userId
     global fBphotoData
     global usersData
     userId=int(spinUserIDVar.get())
     usersData,fBphotoData=fbData.GetPhotoDataById(userId) 
    
#set text to txt2 element
def settxt2(sentence):
     txt2.insert(END, sentence) 

#Use to detect one image by photo ID
def DetectOneImage():
     photo=spinPhotoVar.get()
     user=spinUserVar.get()
     settxt2('User :'+ user)
     settxt2('Photo :'+ photo)
     cgfc.AnalyseOnePhoto(int(photo),int(user))
     #cv2.waitKey(0)     

#Get CSV file Data gathered from cloth detection
def AccessFile():
     global photoData  
     global gender 
     name=usersData['Name'].iloc[0]
     gender=usersData['Gender'].iloc[0]
     photoData=pd.read_csv('FacebookData/' +name+'_FBData_results.csv')       
     DetectImages()        

#Use cloth detection
def DetectCloths():
     global photoData   
     photoData=cgfc.analysePhotoCollection(userId)  
     #DetectImages()      

#Post processing gathered data
def DetectImages(): 
    global PPdata
    PPdata=pd.DataFrame()
    min_thresh=0.3
    
    #txt2.insert(END, photoData) 
    PPdata=DataPreprocessing.DataPreprocessing(photoData,min_thresh)
    #txt2.insert(END, data) 

    
    txt2.insert(END, 'Detection Complete.....')  
    txt2.insert(END, '#################################################')
    txt2.insert(END, '#################################################')

    #topTable -DetectedData
    DDtopWindow = tk.Toplevel()
    DDtopWindowlabel = tk.Label(DDtopWindow, text="Detected Data")
    DDtopWindow.geometry("1024x720")
    table = DDpt = Table(DDtopWindow, dataframe=photoData,showtoolbar=True, showstatusbar=True)
    #DDpt.grid(column=0,row=5)
    #table.grid(column=0,row=5)                                  
    DDpt.show()

    #topTable -PreprocessData
    PDtopWindow = tk.Toplevel()
    PDtopWindowlabel = tk.Label(PDtopWindow, text="Preprocessed Data")
    PDtopWindow.geometry("1024x720")
    table = PPpt = Table(PDtopWindow, dataframe=PPdata,showtoolbar=True, showstatusbar=True)
    #PPpt.grid(column=0,row=5)
    #table.grid(column=0,row=5)                                  
    PPpt.show()


#Use apriori algorithem to find associations
def Apriori():
    
    final_result,fRcolumns=Apriori_Cal.Apriori_Cal_Run(PPdata)
    
    columns=fRcolumns.tolist()
    global ApriotResults
    ApriotResults=final_result
    columns.remove('Photo_ID')
    columns.append('None')
    item1Combo.config(values=columns)
    item2Combo.config(values=columns)
    item3Combo.config(values=columns)
    item4Combo.config(values=columns)
    item5Combo.config(values=columns)
    item6Combo.config(values=columns)
    #print(ApriotResults)
    #print(columns)

#Use Jaccard method to Recommend cloths items    
def Jaccard():
    txt3.delete('1.0', END)
    reccom=Jaccard_Recommendation.JaccardRecommendationRun(PPdata,gender) 
    for rec in reccom:
        txt3.insert(END, '# '+rec)
        txt3.insert(END, '\n\n')

#Check Sentence is Positive or Negetive  
def CheckComment():
        CommentResultTxt.delete('1.0', END)
        sentence=txt.get("1.0",END)
        result=PhotoRating.IsNegetiveComment(sentence)
        CommentResultTxt.insert(END, result) 

#Rate Comments
def GetPhotoRating():
    rating=[]
    rating.clear()
    #photoData=pd.read_csv('CGFC_functions\FB_reaction.csv')
    photoData=pd.read_csv('FacebookData\CGFC-Photos.csv', encoding='cp1252')
    reactRating=PhotoRating.RatingByReactions(fBphotoData)
    #userComments = pd.read_csv("CGFC_functions\Fb_comments.csv", encoding='utf-8')
    userComments = pd.read_csv("FacebookData\CGFC-Comments.csv", encoding='cp1252')
    commentRating=PhotoRating.RatingByComments(userComments)
    rating=pd.merge(commentRating,fBphotoData, on="PhotoID",how='right')
    rating['TotalCommentScore']=rating['TotalCommentScore'].fillna(0)
    rating['total']=rating['total']+rating['TotalCommentScore']

    rating['total']=(rating['total']-rating['total'].min())/(rating['total'].max()-rating['total'].min())
        
 
    #rating=pd.merge(reactRating, rating, on="PhotoID",how='right')




    #txt.insert(END, rating) \

    #topTable -Image Ratings
    RatingWindow = tk.Toplevel()
    RatingWindowlabel = tk.Label(RatingWindow, text="Image Ratings")
    RatingWindow.geometry("1024x720")
    table = Ratingpt = Table(RatingWindow, dataframe=rating,showtoolbar=True, showstatusbar=True)
    #Ratingpt.grid(column=0,row=5)
    #table.grid(column=0,row=5)                                  
    Ratingpt.show()    

def StartClothDetection():
    print("HI")



def item1Selected(event):
     print('item1',item1Combo_val.get())
def item2Selected(event):
     print('item2',item2Combo_val.get())
def item3Selected(event):
     print('item3',item3Combo_val.get())
def item4Selected(event):
     print('item4',item4Combo_val.get())
def item5Selected(event):
     print('item5',item5Combo_val.get())
def item6Selected(event):
     print('item6',item6Combo_val.get())




#Check associations
def checkAssociations():
        ItemsAttributes=[]
        ItemsAttributes.clear()
    
        
        ItemsAttributes.append(item1Combo_val.get())
        ItemsAttributes.append(item2Combo_val.get())
        ItemsAttributes.append(item3Combo_val.get())
        ItemsAttributes.append(item4Combo_val.get())
        ItemsAttributes.append(item5Combo_val.get())
        ItemsAttributes.append(item6Combo_val.get()) 

        
        item=ItemsAttributes.copy()
        for i in ItemsAttributes:
            if((i=='Select')or(i=='None')):
                #print(i)
                item.remove(i)
        ItemsAttributes=item.copy()
    
        print(ItemsAttributes)
        
        AssociationList=Apriori_Cal.CheckAssociationRules(ItemsAttributes,ApriotResults)  
        print(AssociationList)
           
        

        #topTable -Associations
        AssoWindow = tk.Toplevel()
        AssoWindowlabel = tk.Label(AssoWindow, text="Associations Data")
        AssoWindow.geometry("1024x720")
        assoText = scrolledtext.ScrolledText(AssoWindow,width=130,height=50)
        assoText.delete('1.0', END) 
        if len(AssociationList)==0:
                assoText.insert(END, '....NO Associations......')
        else:        
                for i in AssociationList:
                        for item in i['itemset']:
                                assoText.insert(END, (str(item)+' ')) 
                        assoText.insert(END, '\n')
                        assoText.insert(END, '********************************')  
                        assoText.insert(END, '\n') 
        assoText.grid(column=2,row=5)
        #table = Assopt = Table(AssoWindow, dataframe=AssociationList,showtoolbar=True, showstatusbar=True)
        #PPpt.grid(column=0,row=5)
        #table.grid(column=0,row=5)                                  
        #Assopt.show()                

#def generate_new_window():
    


window = Tk()
 
window.title("Crystal Gazor")
 


tab_control = ttk.Notebook(window)
 
#Tab1
############################################# 



tab1 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Photo Rating')

spinUserIDVar=StringVar()
UserIDLAble = Label(tab1, text= 'Set UserID :')
UserIDLAble.grid(column=0, row=1,sticky=W)
spinUserID = Spinbox(tab1, from_=1, to=2,textvariable = spinUserIDVar ,width=5)
spinUserID.grid(column=1,row=1,sticky=W)
btnUserID = Button(tab1, text="Set User ID",command=SetUser)
btnUserID.grid(column=2, row=1 ,sticky=W)


lbl1 = Label(tab1, text= 'Photo Rating')
lbl1.grid(column=0, row=0, sticky=W)

btn = Button(tab1, text="Rating",command=GetPhotoRating)
btn.grid(column=3, row=1 ,sticky=E)
lbl1Sentence = Label(tab1, text= 'Enter Sentence')
lbl1Sentence.grid(column=1, row=3,sticky=W)
txt = scrolledtext.ScrolledText(tab1,width=70,height=10)
txt.grid(column=1,row=4,sticky=W+E+N+S,columnspan=3)

lbl1Result = Label(tab1, text= 'Result')
lbl1Result.grid(column=1, row=5,sticky=W)

CommentResultTxt = scrolledtext.ScrolledText(tab1,width=80,height=2)
CommentResultTxt.grid(column=1,row=6,columnspan=3)
checkbtn = Button(tab1, text="Check Sentence",command=CheckComment)
checkbtn.grid(column=1, row=7,sticky=W+S+N+E,pady=20)
#btn = Button(tab1, text="Click Me", command=clicked)





#Tab2
################################################## 
tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='Image Processing')
lbl2 = Label(tab2, text= 'Image Processing')
lbl2.grid(column=0, row=0,sticky=W)
#btn = Button(tab1, text="Click Me", )

DetectLable = Label(tab2, text= 'Use Cloth Detection -')
DetectLable.grid(column=3, row=1,sticky=W)
btn2 = Button(tab2, text="Detect Cloths",command=DetectCloths)
btn2.grid(column=3, row=1,sticky=E,padx=10, pady=10)

FileLable = Label(tab2, text= 'Use Saved data -')
FileLable.grid(column=3, row=2,sticky=W)
AccessFilebtn = Button(tab2, text="Access Result",command=AccessFile)
AccessFilebtn.grid(column=3, row=2,sticky=E,padx=10, pady=10)
txt2 = scrolledtext.ScrolledText(tab2,width=70,height=5)
txt2.grid(column=1,row=3,columnspan=3, pady=20)


spinUserVar=StringVar()
userNO = Label(tab2, text= 'User NO:')
userNO.grid(column=1, row=4,sticky=E)
spinUser = Spinbox(tab2, from_=1, to=2,textvariable = spinUserVar ,width=5)
spinUser.grid(column=2,row=4,sticky=W)
spinPhotoVar=StringVar()
photoNO = Label(tab2, text= 'PhotoID :')
photoNO.grid(column=2, row=4,sticky=E)
spinPhoto = Spinbox(tab2, from_=0, to=80,textvariable = spinPhotoVar ,width=5)
spinPhoto.grid(column=3,row=4,sticky=W)

DetectOneImagebtn2 = Button(tab2, text="Detect one Cloth",command=DetectOneImage)
DetectOneImagebtn2.grid(column=2, row=9,sticky=E+W+S+N,padx=20, pady=20)





"""

"""

#Tab3
################################################# 
tab3 = ttk.Frame(tab_control)
tab_control.add(tab3, text='Recommendation')
lbl3 = Label(tab3, text= 'Recommendation')
lbl3.grid(column=0, row=0,sticky=W)
btn3 = Button(tab3, text="Recommend Cloths", command=Jaccard)
#btn3 = Button(tab3, text="Click Me",command=Jaccard())
btn3.grid(column=1, row=1,sticky=E,pady=10)
txt3 = scrolledtext.ScrolledText(tab3,width=70,height=10)
txt3.grid(column=1,row=3,pady=20)
 
 
#Tab4
################################################# 
tab4 = ttk.Frame(tab_control)
tab_control.add(tab4, text='Association')
lbl4 = Label(tab4, text='Association')
lbl4.grid(column=0, row=0,sticky=W,padx=20)
AprioriBtn = Button(tab4, text="Initialize", command=Apriori)
AprioriBtn.grid(column=4, row=1,sticky=E,pady=40)



#--Item1 combo
item1Lable = Label(tab4, text= 'Item-1')
item1Lable.grid(column=2, row=2,padx=20)
item1Combo_val = StringVar()
item1Combo = ttk.Combobox(tab4,textvariable = item1Combo_val, values=columns)
item1Combo.grid(column=2, row=3,padx=20)
item1Combo.current(1)
item1Combo.bind("<<ComboboxSelected>>", item1Selected)

#--Item2 combo
item2Lable = Label(tab4, text= 'Item-2')
item2Lable.grid(column=3, row=2,padx=20)
item2Combo_val = StringVar()
item2Combo = ttk.Combobox(tab4,textvariable = item2Combo_val, values=columns)
item2Combo.grid(column=3, row=3,padx=20)
item2Combo.current(1)
item2Combo.bind("<<ComboboxSelected>>", item2Selected)

#--Item3 combo
item2Lable = Label(tab4, text= 'Item-3')
item2Lable.grid(column=4, row=2,padx=20)
item3Combo_val = StringVar()
item3Combo = ttk.Combobox(tab4,textvariable = item3Combo_val, values=columns)
item3Combo.grid(column=4, row=3,padx=20)
item3Combo.current(1)
item3Combo.bind("<<ComboboxSelected>>", item3Selected)

#--Item4 combo
item2Lable = Label(tab4, text= 'Item-4')
item2Lable.grid(column=2, row=8,padx=20)
item4Combo_val = StringVar()
item4Combo = ttk.Combobox(tab4,textvariable = item4Combo_val, values=columns)
item4Combo.grid(column=2, row=9,padx=20)
item4Combo.current(1)
item4Combo.bind("<<ComboboxSelected>>", item4Selected)

#--Item5 combo
item2Lable = Label(tab4, text= 'Item-5')
item2Lable.grid(column=3, row=8,padx=20)
item5Combo_val = StringVar()
item5Combo = ttk.Combobox(tab4,textvariable = item5Combo_val, values=columns)
item5Combo.grid(column=3, row=9,padx=20)
item5Combo.current(1)
item5Combo.bind("<<ComboboxSelected>>", item5Selected)

#--Item6 combo
item2Lable = Label(tab4, text= 'Item-6')
item2Lable.grid(column=4, row=8,padx=20)
item6Combo_val = StringVar()
item6Combo = ttk.Combobox(tab4,textvariable = item6Combo_val, values=columns)
item6Combo.grid(column=4, row=9,padx=20)
item6Combo.current(1)
item6Combo.bind("<<ComboboxSelected>>", item6Selected)


Empty = Label(tab4, text= ' ')
Empty.grid(column=3, row=7,sticky=W+E+S+N,pady=20)
checkAssoBtn = Button(tab4, text="check Associations", command=checkAssociations)
checkAssoBtn.grid(column=3, row=10,sticky=W+E+S+N,pady=40)


################################################# 
tab_control.pack(expand=1, fill='both')
 


    


def doSomething():
    # check if saving
    # if not:
    window.destroy()








window.protocol('WM_DELETE_WINDOW', doSomething)  # root is your root window



window.geometry("780x420")
 
window.mainloop()