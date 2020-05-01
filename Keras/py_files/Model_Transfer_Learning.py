from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, GlobalAveragePooling2D
from keras import backend as K 

def MODEL(num_classes, img_width = 224, img_height = 224):
    if K.image_data_format() == 'channels_first': 
        input_shape = (3, img_width, img_height) 
    else: 
        input_shape = (img_width, img_height, 3) 

    model = Sequential()
    model.add(MobileNetV2(input_shape = input_shape, include_top=False, weights='imagenet'))

    model.add(GlobalAveragePooling2D())
    
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax')) 
    return model