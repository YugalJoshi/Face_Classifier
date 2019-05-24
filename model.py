"""Created on  May  19 09:14:59 2019"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import numpy as np
import os
#import theano
from PIL import Image
from numpy import *
# SKLEARN

from sklearn.utils import shuffle
#from sklearn import cross_validation

# input image dimensions
img_rows, img_cols = 64, 64

# number of channels
img_channels = 1

# all images are already processed 
path2 = 'D:\\processed_image'  #path of folder of iamges    

goingin = os.listdir(path2)
num_samples=size(goingin)
immatrix = []

im1 = array(Image.open(path2+'\\'+ goingin[0])) # open one image to get size
immatrix.append(im1)
immatrix = np.array(immatrix).flatten()

immatrix = array([array(Image.open(path2+'\\'+ pic)).flatten() for pic in goingin] ,'f')
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(goingin) #get the number of images
#print(imnbr)   
#print(immatrix)
 
 #labelling of data
label=np.ones((num_samples,),dtype = int) 
label[0:676]=0
label[676:1274]=1

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]
print(train_data) 

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 2


(X, y) = (train_data[0],train_data[1])

# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

print(X_train.shape[0])
X_train = X_train.reshape(X_train.shape[0],  img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0],  img_rows, img_cols,1 )

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print (train_data[0].shape)
print (train_data[1].shape)


#%%

print(X_test.shape[0], 'test samples')

#convert class vectors to binary class matrices   && one hot encoding 
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



#%%
def Create_model():
    model = Sequential()
    model.add(Convolution2D(64, 3, 3,
                        border_mode='valid',
                        input_shape=(img_rows, img_cols,1)))
    model.add(BatchNormalization())  # batchnormalization , take care of spelling error (s and z)
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Convolution2D(32, 3, 3))
    model.add(BatchNormalization())
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3,3), activation='relu')) 
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (4,4), activation='relu')) 
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

model = Create_model()
hist = model.fit(X_train, Y_train, batch_size=32, epochs=1
                 , verbose=1, validation_split=0.2)
      
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# to save model
save_weight = "weightsCNN.h5"
model.save_weights(save_weight,overwrite=True)
