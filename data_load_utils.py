import numpy as np
from skimage import io
import os
import tensorflow as tf

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

batchnumber=0
batchsize=10
def nextbatch(x_in, y_in, batchnumber=batchnumber, batchsize=batchsize):
    startindex=batchnumber*batchsize
    stopindex=startindex+batchsize
    batchnumber+=1
    return x_in[startindex:stopindex], y_in[startindex:stopindex]

def tensorreshape(x_in, y_in, batch_size, width, height, nchannels):
    x_in=np.reshape(x_in, (batch_size, height, width, nchannels))
    y_in=np.reshape(y_in, (batch_size, height, width, 1))
    reshaped_image_x = tf.cast(x_in, tf.float32)
    xr=tf.reshape(reshaped_image_x, shape=(batch_size, width, height, nchannels))
    reshaped_image_y = tf.cast(y_in, tf.float32)
    yr=tf.reshape(reshaped_image_y, shape=(batch_size, width, height, 1))
    xrc=tf.image.resize_image_with_crop_or_pad(xr, 320,320)
    yrc=tf.image.resize_image_with_crop_or_pad(yr, 320,320)
    return xrc, yrc

