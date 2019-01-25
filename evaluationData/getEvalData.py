import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def GetObjectByClass(className):
    classItems = pd.read_csv('evaluationData/'+className+'_labels.csv')
    #classItems = pd.read_csv(className+'_labels.csv')
    return classItems


