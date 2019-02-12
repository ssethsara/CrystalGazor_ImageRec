from tkinter import *
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from tkinter.ttk import *
from pandastable import Table, TableModel
import cv2

import pandas as pd
import numpy as np

from CGFC_functions import CGFCConfig
from CGFC_functions import DataPreprocessing
from CGFC_functions import Jaccard_Recommendation
from CGFC_functions import Apriori_Cal
from CGFC_functions import PhotoRating
import CGFC as cgfc


min_thresh=0.3
columns=['None','Select']
ApriotResults=pd.DataFrame()


PPdata=pd.DataFrame()
userData=pd.DataFrame()

def settxt2(sentence):
     txt2.insert(END, sentence) 

def DetectOneImage():
     photo=spinPhotoVar.get()
     user=spinUserVar.get()
     settxt2('User :'+ user)
     settxt2('Photo :'+ photo)
     cgfc.AnalyseOnePhoto(int(photo),int(user))
     #cv2.waitKey(0)     


def AccessFile():
     global userData   
     userData=pd.read_csv('FacebookData\Supun Sethsara_FBData_results.csv')       
     DetectImages()        

def DetectCloths():
     global userData   
     userData=cgfc.analysePhotoCollection(1)  
     #DetectImages()     

def DetectImages(): 
    global PPdata
    PPdata=pd.DataFrame()
    min_thresh=0.3
    
    #txt2.insert(END, userData) 
    PPdata=DataPreprocessing.DataPreprocessing(userData,min_thresh)
    #txt2.insert(END, data) 

    
    txt2.insert(END, 'Detection Complete.....')  
    txt2.insert(END, '#################################################')
    txt2.insert(END, '#################################################')

    #topTable -DetectedData
    DDtopWindow = tk.Toplevel()
    DDtopWindowlabel = tk.Label(DDtopWindow, text="Detected Data")
    DDtopWindow.geometry("1024x720")
    table = DDpt = Table(DDtopWindow, dataframe=userData,showtoolbar=True, showstatusbar=True)
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

    
def Jaccard():
    txt3.delete('1.0', END)
    reccom=Jaccard_Recommendation.JaccardRecommendationRun(PPdata) 
    txt3.insert(END, reccom)
  


def GetPhotoRating():
    rating=[]
    rating.clear()
    txt.delete('1.0', END)
    userReactions=pd.read_csv('CGFC_functions\FB_reaction.csv')
    reactRating=PhotoRating.RatingByReactions(userReactions)
    userComments = pd.read_csv("CGFC_functions\Fb_comments.csv", encoding='utf-8')
    commentRating=PhotoRating.RatingByComments(userComments)
    rating=pd.merge(reactRating, commentRating, on="PhotoID")
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
        assoText = scrolledtext.ScrolledText(AssoWindow,width=120,height=50)
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
 
dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))



tab_control = ttk.Notebook(window)
 
#Tab1
############################################# 
tab1 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Photo Rating')
lbl1 = Label(tab1, text= 'Photo Rating')
lbl1.grid(column=0, row=0)
#btn = Button(tab1, text="Click Me", command=clicked)
btn = Button(tab1, text="Rating",command=GetPhotoRating)
btn.grid(column=0, row=1)
txt = scrolledtext.ScrolledText(tab1,width=95,height=20)
txt.grid(column=1,row=2)



#Tab2
################################################## 
tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='Image Processing')
lbl2 = Label(tab2, text= 'Image Processing')
lbl2.grid(column=0, row=0)
#btn = Button(tab1, text="Click Me", )
btn2 = Button(tab2, text="Detect Cloths",command=DetectCloths)
btn2.grid(column=0, row=1)
AccessFilebtn = Button(tab2, text="Access Result",command=AccessFile)
AccessFilebtn.grid(column=0, row=2)
txt2 = scrolledtext.ScrolledText(tab2,width=95,height=10)
txt2.grid(column=1,row=3)
DetectOneImagebtn2 = Button(tab2, text="Detect one Cloth",command=DetectOneImage)
DetectOneImagebtn2.grid(column=0, row=4)

spinUserVar=StringVar()
userNO = Label(tab2, text= 'User NO:')
userNO.grid(column=0, row=5)
spinUser = Spinbox(tab2, from_=0, to=1,textvariable = spinUserVar ,width=5)
spinUser.grid(column=0,row=6)
spinPhotoVar=StringVar()
photoNO = Label(tab2, text= 'PhotoID :')
photoNO.grid(column=0, row=7)
spinPhoto = Spinbox(tab2, from_=0, to=30,textvariable = spinPhotoVar ,width=5)
spinPhoto.grid(column=0,row=8)






"""

"""

#Tab3
################################################# 
tab3 = ttk.Frame(tab_control)
tab_control.add(tab3, text='Recommendation')
lbl3 = Label(tab3, text= 'Jaccard')
lbl3.grid(column=0, row=0)
btn3 = Button(tab3, text="Recommend Cloths", command=Jaccard)
#btn3 = Button(tab3, text="Click Me",command=Jaccard())
btn3.grid(column=0, row=1)
txt3 = scrolledtext.ScrolledText(tab3,width=95,height=20)
txt3.grid(column=1,row=2)
 
 
#Tab4
################################################# 
tab4 = ttk.Frame(tab_control)
tab_control.add(tab4, text='Association')
lbl4 = Label(tab4, text='Apriori')
lbl4.grid(column=0, row=0)
AprioriBtn = Button(tab4, text="Initialize", command=Apriori)
AprioriBtn.grid(column=0, row=1)



#--Item1 combo
item1Lable = Label(tab4, text= 'Item-1')
item1Lable.grid(column=0, row=2)
item1Combo_val = StringVar()
item1Combo = ttk.Combobox(tab4,textvariable = item1Combo_val, values=columns)
item1Combo.grid(column=0, row=3)
item1Combo.current(1)
item1Combo.bind("<<ComboboxSelected>>", item1Selected)

#--Item2 combo
item2Lable = Label(tab4, text= 'Item-2')
item2Lable.grid(column=1, row=2)
item2Combo_val = StringVar()
item2Combo = ttk.Combobox(tab4,textvariable = item2Combo_val, values=columns)
item2Combo.grid(column=1, row=3)
item2Combo.current(1)
item2Combo.bind("<<ComboboxSelected>>", item2Selected)

#--Item3 combo
item2Lable = Label(tab4, text= 'Item-3')
item2Lable.grid(column=2, row=2)
item3Combo_val = StringVar()
item3Combo = ttk.Combobox(tab4,textvariable = item3Combo_val, values=columns)
item3Combo.grid(column=2, row=3)
item3Combo.current(1)
item3Combo.bind("<<ComboboxSelected>>", item3Selected)

#--Item4 combo
item2Lable = Label(tab4, text= 'Item-4')
item2Lable.grid(column=3, row=2)
item4Combo_val = StringVar()
item4Combo = ttk.Combobox(tab4,textvariable = item4Combo_val, values=columns)
item4Combo.grid(column=3, row=3)
item4Combo.current(1)
item4Combo.bind("<<ComboboxSelected>>", item4Selected)

#--Item5 combo
item2Lable = Label(tab4, text= 'Item-5')
item2Lable.grid(column=4, row=2)
item5Combo_val = StringVar()
item5Combo = ttk.Combobox(tab4,textvariable = item5Combo_val, values=columns)
item5Combo.grid(column=4, row=3)
item5Combo.current(1)
item5Combo.bind("<<ComboboxSelected>>", item5Selected)

#--Item6 combo
item2Lable = Label(tab4, text= 'Item-6')
item2Lable.grid(column=5, row=2)
item6Combo_val = StringVar()
item6Combo = ttk.Combobox(tab4,textvariable = item6Combo_val, values=columns)
item6Combo.grid(column=5, row=3)
item6Combo.current(1)
item6Combo.bind("<<ComboboxSelected>>", item6Selected)



checkAssoBtn = Button(tab4, text="check Associations", command=checkAssociations)
checkAssoBtn.grid(column=6, row=3)


################################################# 
tab_control.pack(expand=1, fill='both')
 


    


def doSomething():
    # check if saving
    # if not:
    window.destroy()








window.protocol('WM_DELETE_WINDOW', doSomething)  # root is your root window



window.geometry("1024x480")
 
window.mainloop()