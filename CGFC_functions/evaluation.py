import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature


#Used this to evaluate the cloth detection model
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou=0.0
    iou = iou+(intersection_area / float(bb1_area + bb2_area - intersection_area))
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def getTPCount(evalData,min_IoU):
    evalData= evalData.loc[(evalData['loc_Accuracy'] >= min_IoU)&(evalData['DetectedClass'] == evalData['class'])]
    return len(evalData)

def getFPCount(evalData):
    evalData=evalData.loc[(evalData['DetectedClass'] != evalData['class']) &(evalData['DetectedClass'] !='UnDetected')]
    return len(evalData)

def getFNCount(evalData):
    evalData=evalData.loc[evalData['DetectedClass'] =='UnDetected']
    return len(evalData)  

def getPrecision(tp,fp):
    if (tp+fp)==0:
        return 0
    return tp/(tp+fp)

def getRecall(tp,fn):
    if (tp+fn)==0:
        return 0
    return tp/(tp+fn)

def ListOnTreshold(evalResults,min_IoU,min_tresh_values):
    resultsPR=pd.DataFrame()
    for min_tresh in min_tresh_values:
        evalResultsFiltered=evalResults.loc[evalResults['min_tresh']==min_tresh]
        evalResultsFiltered.sort_values('Confidence')
        tp=getTPCount(evalResultsFiltered,min_IoU)
        fp=getFPCount(evalResultsFiltered)
        fn=getFNCount(evalResultsFiltered)

        wholeTestDataCount=len(evalResultsFiltered)
        PredictionSuccess=tp/wholeTestDataCount

        precision=getPrecision(tp,fp)
        recall=getRecall(tp,fn)
        resultsPR =resultsPR.append(pd.DataFrame({'min_tresh_values' : [min_tresh],
            'min_IoU' : [min_IoU],
            'precision' : [precision],
            'recall' : [recall],
            'SuccessRate' : [PredictionSuccess],
            'TP' : [tp],
            'FP' : [fp],
            'FN' : [fn] }))
    
    return resultsPR


def averagePrecision(resultsPR):
    totalPrecision=0
    #resultsPR=resultsPR.loc[resultsPR['SuccessRate']>0.5]
    totalPrecision=resultsPR['precision'].sum()
    return totalPrecision/len(resultsPR)



def vizualize(resultsPR,average_precision):
    precision=resultsPR['precision'].values
    recall=resultsPR['recall'].values
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
    plt.step(recall, precision, color='b', alpha=0.2,
            where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
    plt.show()        


def evaluate(evalResults):
    
    min_IoU=0.5
    min_tresh_values=[0.2,0.4,0.6,0.8]
    #evalResults = pd.read_csv('evaluationData/Evaluation_results.csv')
    ResultsPandR=ListOnTreshold(evalResults,min_IoU,min_tresh_values)
    ap=averagePrecision(ResultsPandR)
   
   
    print(ResultsPandR)
    print('average Precision : ', ap)
    #print('Mean average Precision : ', mAp)
    #vizualize(ResultsPandR,ap)

"""
os.chdir("..")
evalResults = pd.read_csv('evaluationData/Evaluation_results.csv')

evaluate(evalResults)"""
