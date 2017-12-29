import numpy as np
import PIL.Image
import glob
import os
import re

ADEDIR=""


def onlyhuman(imagein):
    imagein[imagein!=13]=0
    imagein[image==13]=1
    return imagein
    
def parsepaths(settype): # settype is either string "training" or "validation"
    # Load image and notations in separate lists
    annot=sorted(glob.glob(ADEDIR+"annotations/"+settype+"/*.png"))
    n=[re.findall(r"(ADE_[A-z0-9_]*).png",i)[0] for i in annot]
    return zip(annot, n)

def allconvert(ADEDIR)
    training=parsepaths("training")
    validation=parsepaths("validation")

    if not os.path.exists(ADEDIR+"annotations/converted_training/"):
        os.makedirs(ADEDIR+"annotations/converted_training/")
    else:
        return
    if not os.path.exists(ADEDIR+"annotations/converted_validation/"):
        os.makedirs(ADEDIR+"annotations/converted_validation/")

    for i,j in training:
        image = PIL.Image.open(i)
        image = np.array(image)
        out=onlyhuman(image)
        im = PIL.Image.fromarray(out)
        im.save(ADEDIR+"annotations/converted_training/"+j+".png")    

    for i,j in validation:
        image = PIL.Image.open(i)
        image = np.array(image)
        out=onlyhuman(image)
        im = PIL.Image.fromarray(out)
        im.save(ADEDIR+"annotations/converted_validation/"+j+".png")