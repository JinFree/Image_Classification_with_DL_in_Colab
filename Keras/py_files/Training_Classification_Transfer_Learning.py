import os
import tensorflow as tf
import keras
import json
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from Model_Transfer_Learning import MODEL as MODEL_TRANSFER

def main(_root_dir = "/content/cats_and_dogs_filtered"
         , _epochs = 50, _batch_size = 16):
    img_width, img_height = 224, 224
    EPOCHS     = 5
    BATCH_SIZE = 16

    train_datagen = ImageDataGenerator( rescale = 1. / 255, rotation_range = 15) 
    test_datagen = ImageDataGenerator( rescale = 1. / 255 )

    train_data_dir = _root_dir + "/train"
    validation_data_dir = _root_dir + "/validation"

    label_list = os.listdir(train_data_dir)
    label_list.sort() 

    train_generator = train_datagen.flow_from_directory(
        train_data_dir
        , target_size = (img_width, img_height)
        , shuffle = True
        , batch_size = BATCH_SIZE
        , classes = label_list
        , class_mode ='categorical') 

    validation_generator = test_datagen.flow_from_directory( 
        validation_data_dir
        , target_size = (img_width, img_height)
        , shuffle = True
        , batch_size = BATCH_SIZE
        , classes = label_list
        , class_mode ='categorical') 

    f = open("label_map.txt", 'w')
    for i in range(len(label_list)):
        f.write(label_list[i]+'\n')
    f.close()    
    num_classes = len(label_list)
    
    model = MODEL_TRANSFER(num_classes)
    model_str = "Keras_Classification_Model_Trnasfer"

    optimizer = optimizers.SGD(lr=0.0001)
    model.compile(optimizer = optimizer
                , loss = 'categorical_crossentropy'
                , metrics = ['accuracy'])


    model.fit_generator(generator = train_generator
                        , steps_per_epoch = train_generator.n//train_generator.batch_size
                        , epochs = EPOCHS
                        , validation_data = validation_generator
                        , validation_steps = validation_generator.n//validation_generator.batch_size)


    model_json = model.to_json()
    with open(model_str + ".json", "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights(model_str + ".h5")

if __name__ == "__main__":
    main("/content/cats_and_dogs_filtered", 50, 16)