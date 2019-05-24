"""
Created on May 17 20:42:25 2019
@author: Yugal Joshi
"""
import os
from PIL import Image

path = "D:\\AI-ML projects\\images\\training set\\images"  #path to the folder that contain all images.

for i_am_in in os.listdir(path):

    img = Image.open(path +'\\' + i_am_in)   # using os.listdir (listing the directory), we are inside the folder "images" and Image.open will give the array of image
    img = img.convert('L')  # For constraints of computation power, images are converted from RGB to gray scale
    img = img.resize((64,64)) # resizing of images
    img.save(path+'\\'+i_am_in,"JPEG")  # And hope u got .save() method.....
