from PIL import Image, ImageChops
import numpy as np
from math import sqrt
import os
import argparse

parser = argparse.ArgumentParser(description="First test task on images similarity.")
parser.add_argument('--path', help = 'folder with images', required=True)
args = parser.parse_args()
path = args.path
images = os.listdir(path)

#Detect duplicate (images which are exactly the same)
def detectDuplicate(image1, image2):
    if np.asarray(image1).shape == np.asarray(image2).shape:
        difValue = np.sum( np.asarray(image1) - np.asarray(image2) )
        if difValue == 0:
            return True
    return False

#Convolve image to fix size - create feature
def convolve(image):
    return (np.array( image
                 .convert('L') # convert to grayscale using PIL            
                 .resize( (32, 32), resample=Image.BICUBIC) ) # reduce size and smooth a bit using PIL
                 ).astype(np.int)

#Detect modification (images which differ by size, blur level or noise filters)
def detectModification(image1, image2):
    difValue = np.abs( convolve(image1) - convolve(image2) ).sum() / 1024 # average of all pixels of feature differences, 32 * 32 = 1024
    if difValue < 2: # Let's assume that all images with difValue less 2 are modified to each other
        return True
    return False

#Detect similar (images of the same scene from another angle)
def detectSimilar(image1, image2):
    #The histogram is returned as a list of pixel counts, 
    #one for each pixel value in the source image. If the image has more than one band, 
    #the histograms for all bands are concatenated (for example, the histogram for an “RGB” image contains 768 values).
    hist1, hist2 = image1.histogram(), image2.histogram()
    
    difValue = np.abs( convolve(image1) - convolve(image2) ).sum() / 1024 # average of all pixels of feature differences, 32 * 32 = 1024

    RMS = sqrt( sum( map(lambda x, y: (x - y)**2, hist1, hist2) ) / len(hist1) )
    
    if RMS < 3072 and difValue < 32: # Let's assume that all images with RMS less 3072 and difValue less 32 are similar to each other
        return True
    return False


for i in range(0, len(images) - 1):
    for j in range(i + 1, len(images)):
        image1 = Image.open(path + os.path.sep + images[i])
        image2 = Image.open(path + os.path.sep + images[j])
        if detectDuplicate(image1, image2):
            print(images[i], "\t", images[j], "\t are duplicate")
            continue
        if detectModification(image1, image2):
            print(images[i], "\t", images[j], "\t are modification")
            continue
        if detectSimilar(image1, image2):
            print(images[i], "\t", images[j], "\t are similar")
            continue
