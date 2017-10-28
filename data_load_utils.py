import numpy as np
from skimage import io
import os

# Define data import paths
# Download the dataset and extract the CamVid folder to a location of your choosing
# Take note that you might have to adjust the following depending on your OS
path_separator="\\" # How your OS separates paths
data_root="..\\CamVid" # Where you unzipped the CamVid dataset
test_paths=data_root+path_separator+"test.txt"
train_paths=data_root+path_separator+"train.txt"
valid_paths=data_root+path_separator+"valid.txt"

# Data import utils
def parsepaths(location):
    # Load image and notations in separate lists
    x=[]
    y=[]
    with open(location) as f:
        data = f.read()
    f.closed
    for l in data.split("\n"):
        if len(l)>17: # If the line contains more characters than the typical filename of one sample image
            x.append(path_replace(l.split(" ")[0]))
            y.append(path_replace(l.split(" ")[1]))
    return x, y

def path_replace(path):
    path=path.replace("/SegNet/CamVid/","../")
    path=path.replace("/",path_separator)
    path=path.replace("..", data_root)
    return path

def loadimages(x_image,y_image):
    #Images to numpy arrays
    x=[]
    y=[]
    for image in x_image:
        x.append(io.imread(image))
    for image in y_image:
        y.append(io.imread(image))
    return x, y

# To load the  test images into x and y, do the following:	
#    locx, locy = parsepaths(test_paths)
#    x, y = loadimages(locx, locy)