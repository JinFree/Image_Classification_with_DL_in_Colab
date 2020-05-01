import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
import json

def restoreModel(model_str):
    with open(model_str + ".json", "r") as f:
        model_json = json.load(f)
    model_restored = model_from_json(model_json)
    model_restored.load_weights(model_str + ".h5")
    model_restored.summary()
    return model_restored

def load_label_map(textFile):
    return np.loadtxt(textFile, str, delimiter='\t')
    
def cv_image_read(image_path):
    print(image_path)
    return cv2.imread(image_path)

def print_result(inference_result, class_map):
    class_text = class_map[np.argmax(inference_result)]
    print(inference_result)
    print(class_text)

def inference_image(opencv_image, model, size = (224, 224)):
    image = cv2.resize(opencv_image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = np.expand_dims(image, axis=0)
    result = model.predict(image_tensor)
    return result

def inference_main(image_path, _is_transfer = False):
    img_width, img_height = 224, 224
    class_map = load_label_map('label_map.txt')
    num_classes = len(class_map)
    if not _is_transfer:
        model_str = "Keras_Classification_Model_Scratch"
    else:
        model_str = "Keras_Classification_Model_Trnasfer"
    model = restoreModel(model_str)
    opencv_image = cv_image_read(image_path)
    inference_result = inference_image(opencv_image, model, (img_width, img_height))
    print_result(inference_result, class_map)
    
def main(index, _PATH = "/content/cats_and_dogs_filtered/validation"):
    PATH = _PATH
    validation_cats_dir = PATH + '/cats'  # directory with our validation cat pictures
    validation_dogs_dir = PATH + '/dogs'  # directory with our validation dog pictures
    list_of_test_cats_images = os.listdir(validation_cats_dir)
    list_of_test_dogs_images = os.listdir(validation_dogs_dir)
    for idx in range(len(list_of_test_cats_images)):
        list_of_test_cats_images[idx] = validation_cats_dir + '/'+list_of_test_cats_images[idx]
    for idx in range(len(list_of_test_dogs_images)):
        list_of_test_dogs_images[idx] = validation_dogs_dir + '/'+list_of_test_dogs_images[idx]
    list_of_test_images = list_of_test_cats_images + list_of_test_dogs_images
    inference_main(list_of_test_images[index])
    return list_of_test_images
    
if __name__ == "__main__":
    main(600)