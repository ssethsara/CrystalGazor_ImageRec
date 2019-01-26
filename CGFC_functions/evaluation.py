import pandas as pd
import numpy as np
import os



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
    evalData= evalData.loc[(evalData['loc_Accuracy'] >= min_IoU)]
    return len(evalData)

def getFPCount(evalData):
    evalData=evalData.loc[(evalData['DetectedClass'] != evalData['class']) &(evalData['DetectedClass'] !='UnDetected')]
    return len(evalData)

def getFNCount(evalData):
    evalData=evalData.loc[evalData['DetectedClass'] =='UnDetected']
    return len(evalData)  

def getPrecision(evalData):
    tp=getTPCount(evalData,0.75)
    fp=getFPCount(evalData)
    fn=getFNCount(evalData)
    return tp/(tp+fp)

def getRecall(evalData):
    tp=getTPCount(evalData,0.75)
    fp=getFPCount(evalData)
    fn=getFNCount(evalData)
    return tp/(tp+fn)



def test():
    os.chdir("..")
    evalResults = pd.read_csv('evaluationData/Evaluation_results.csv')
    print('Precision: ',getPrecision(evalResults))
    print('Recall: ',getRecall(evalResults))

test()