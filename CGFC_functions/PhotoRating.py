import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.stem import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
#import matplotlib.pyplot as plt

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


import io

import pkg_resources
#from sklearn.externals import joblib

#nltk.download('stopwords')




def RatingByReactions(userReactions):

    userReactions.drop(['UserId'], axis = 1, inplace = True)
    userReactions.drop(['PostedBy'], axis = 1, inplace = True)
    userReactions.drop(['PhotoURL'], axis = 1, inplace = True)
    userReactions.drop(['UploadedDate'], axis = 1, inplace = True)
    userReactions.drop(['Tag_width'], axis = 1, inplace = True)
    userReactions.drop(['Tag_height'], axis = 1, inplace = True)
    userReactions.drop(['Tag_left'], axis = 1, inplace = True)
    userReactions.drop(['Tag_top'], axis = 1, inplace = True)
    userReactions.drop(['Tagged_user_Count'], axis = 1, inplace = True)
    #userReactions.drop(['PhotoID'], axis = 1, inplace = True)


    userReactions['total'] = userReactions['Love'] *5 + userReactions['Likes'] *4 + userReactions['Wow']*3 + userReactions['Haha']*-1 + userReactions['Sad']*-2 + userReactions['Angry']*-3

    return userReactions

def trainWordIdentifyingModel():
    # Read in data
    data = pd.read_csv("clean_data2.csv", encoding='cp1252')
    texts = data['text'].astype(str)
    y = data['is_offensive']

    #preprocessing
    texts = texts.apply(lambda x: " ".join(x.lower() for x in x.split()))
    texts = texts.str.replace('[^\w\s]','')

     # Vectorize the text
    vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
    X = vectorizer.fit_transform(texts)   

    # Train the model
    model = LinearSVC(C=1.0,class_weight="balanced", dual=False, tol=1e-2, max_iter=100000000)
    cclf = CalibratedClassifierCV(base_estimator=model)
    cclf.fit(X, y)

    # Save the model
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(cclf, 'model.joblib') 

  




def _get_profane_prob(prob):
  return prob[1]

def predict(texts):
  return model.predict(vectorizer.transform(texts))

def predict_prob(texts):
  return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))


def RatingByComments(userComments):
    #Number of words
    userComments['word_count'] = userComments['Comment'].apply(lambda x: len(str(x).split(" ")))
    OnlyComments=userComments[['PhotoId','Comment','word_count']]
    OnlyComments

    #Transfer comments into lower case
    OnlyComments['Comment'] = OnlyComments['Comment'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    #Removing Punctuation
    OnlyComments['Comment'] = OnlyComments['Comment'].str.replace('[^\w\s]','')

  
   
    stop = stopwords.words('english')
    OnlyComments['Comment'] = OnlyComments['Comment'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    OnlyComments['Comment'] 

    #Stemming
    
    st = PorterStemmer()
    CommentScored=pd.DataFrame()
    OnlyComments['CommentScore']=0
    for index,comments in enumerate(OnlyComments['Comment']):
        isPositive=int(predict([comments])[0])
        if isPositive==0:
            CommentScored=CommentScored.append(pd.DataFrame({
                'PhotoId':OnlyComments['PhotoId'][index],
                'Comment':OnlyComments['Comment'][index],
                'CommentScore':[0]
            }))
        elif isPositive==1:
            CommentScored=CommentScored.append(pd.DataFrame({
                'PhotoId':OnlyComments['PhotoId'][index],
                'Comment':OnlyComments['Comment'][index],
                'CommentScore':[1]
            }))
        
    PhotoIDs=list(set(CommentScored['PhotoId'].values))
    PhotoIDs
    CommentScoreSum=pd.DataFrame()
    PhotoIDs=list(set(CommentScored['PhotoId'].values))
    PhotoIDs
    CommentScoreSum=pd.DataFrame()
    for photoid in PhotoIDs:
        score=CommentScored['CommentScore'].loc[CommentScored['PhotoId']==photoid]
        CommentScoreSum=CommentScoreSum.append(pd.DataFrame({
                        'PhotoID':[photoid],
                        'TotalCommentScore':[score.sum()]
                    }))
        
        CommentScoreSum


    return CommentScoreSum




"""
#df = pd.read_csv("Fb_reaction.csv", encoding='cp1252')
userReactions=pd.read_csv('FB_reaction.csv')
data=RatingByReactions(userReactions)
print(data)    
"""

#trainWordIdentifyingModel()
"""
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

userComments = pd.read_csv("Fb_comments.csv", encoding='utf-8')
Rating=RatingByComments(userComments)
print(Rating)
"""

vectorizer = joblib.load('CGFC_functions/vectorizer.joblib')
model = joblib.load('CGFC_functions/model.joblib')