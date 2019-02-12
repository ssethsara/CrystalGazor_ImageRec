# import the necessary packages
import numpy as np
import cv2

import webcolors

# import the necessary packages
from sklearn.cluster import KMeans
import scipy.spatial as sp

import matplotlib.pyplot as plt
import argparse
import utils
import cv2


websafe_colors = [(0,0,0),
                  (255,255,255),
                  (255,0,0),
                  (0,255,0),
                  (0,0,255),
                  (255,255,0),
                  (0,255,255),
                  (255,0,255),
                  (128, 128, 128)
                  ] # list of web-save colors  
  



def centroid_histogram(clt):
    	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	return hist


def plot_colors(hist, centroids):
    	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
 
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	return bar    


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name



def dominant_color_detector(image,amount):

    num_of_clusters=amount

    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    #----image = cv2.imread(args["download.jpg"])

    #image = cv2.imread("test.jpg")

    dominet_colors=[]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape 

    image = cv2.resize(image, (0,0), fx=0.1, fy=0.1) 
    h,w,bpp = np.shape(image)

    for py in range(0,h):
        for px in range(0,w):
            input_color = (image[py][px][0],image[py][px][1],image[py][px][2])
            tree = sp.KDTree(websafe_colors) # creating k-d tree from web-save colors
            ditsance, result = tree.query(input_color) # get Euclidean distance and index of web-save color in tree/list
            nearest_color = websafe_colors[result]
            
            image[py][px][0]=nearest_color[0]
            image[py][px][1]=nearest_color[1]
            image[py][px][2]=nearest_color[2]
    
    #cv2.imshow('matrix', image)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    #-----clt = KMeans(n_clusters = args["clusters"])
    clt = KMeans(n_clusters = num_of_clusters)
    clt.fit(image)
    
    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    #color code and persentage desplay
    for (percent, color) in zip(hist, clt.cluster_centers_):
        colorCode=color.astype("uint8").tolist()
        #print(colorCode," ","Percentage %.2f"%(percent*100),"%")
        #generate color name
        #to RGB round
        input_color = (colorCode[0],colorCode[1],colorCode[2])
        
        tree = sp.KDTree(websafe_colors) # creating k-d tree from web-save colors
        ditsance, result = tree.query(input_color) # get Euclidean distance and index of web-save color in tree/list
        nearest_color = websafe_colors[result]
        actual_name, closest_name = get_colour_name((nearest_color[0],nearest_color[1],nearest_color[2]))
        
        #actual_name, closest_name = get_colour_name((colorCode[0],colorCode[1],colorCode[2]))
        #print("Actual colour name:", actual_name, ", closest colour name:", closest_name) 

        color_data=[colorCode,'%.2f'%(percent*100),actual_name,closest_name]

        dominet_colors.append(color_data)
    
    return dominet_colors
