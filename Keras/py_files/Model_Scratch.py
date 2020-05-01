import keras
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D , BatchNormalization, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 

def MODEL(num_classes, img_width = 224, img_height = 224):
    if K.image_data_format() == 'channels_first': 
        input_shape = (3, img_width, img_height) 
    else: 
        input_shape = (img_width, img_height, 3) 
  
    model = Sequential() 
    model.add(Conv2D(32, (3, 2), input_shape = input_shape)) 
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    
    model.add(Conv2D(64, (3, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    
    model.add(Conv2D(128, (3, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 

    model.add(GlobalAveragePooling2D()) 
    
    model.add(Dense(64)) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(num_classes)) 
    model.add(Activation('softmax')) 
    return model