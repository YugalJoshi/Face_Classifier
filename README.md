# Face_Classifier
This is binary classification problem based on Constitutional Neural  Network.
The dataset contain around 1274 images(64 X 64) of two of my frineds.
The model is trained on 916 samples and validate on 230 samples.
Summary of the model :

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_68 (Conv2D)           (None, 62, 62, 64)        640       
_________________________________________________________________
batch_normalization_1 (Batch (None, 62, 62, 64)        256       
_________________________________________________________________
activation_1 (Activation)    (None, 62, 62, 64)        0         
_________________________________________________________________
conv2d_69 (Conv2D)           (None, 60, 60, 32)        18464     
_________________________________________________________________
batch_normalization_2 (Batch (None, 60, 60, 32)        128       
_________________________________________________________________
activation_2 (Activation)    (None, 60, 60, 32)        0         
_________________________________________________________________
max_pooling2d_49 (MaxPooling (None, 30, 30, 32)        0         
_________________________________________________________________
conv2d_70 (Conv2D)           (None, 28, 28, 32)        9248      
_________________________________________________________________
max_pooling2d_50 (MaxPooling (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_71 (Conv2D)           (None, 11, 11, 32)        16416     
_________________________________________________________________
flatten_17 (Flatten)         (None, 3872)              0         
_________________________________________________________________
dense_17 (Dense)             (None, 1024)              3965952   
_________________________________________________________________
batch_normalization_3 (Batch (None, 1024)              4096      
_________________________________________________________________
activation_3 (Activation)    (None, 1024)              0         
_________________________________________________________________
dense_18 (Dense)             (None, 512)               524800    
_________________________________________________________________
batch_normalization_4 (Batch (None, 512)               2048      
_________________________________________________________________
activation_4 (Activation)    (None, 512)               0         
_________________________________________________________________
dense_19 (Dense)             (None, 64)                32832     
_________________________________________________________________
batch_normalization_5 (Batch (None, 64)                256       
_________________________________________________________________
activation_5 (Activation)    (None, 64)                0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_20 (Dense)             (None, 2)                 130       
_________________________________________________________________
activation_6 (Activation)    (None, 2)                 0         
=================================================================
Total params: 4,575,266
Trainable params: 4,571,874
Non-trainable params: 3,392


