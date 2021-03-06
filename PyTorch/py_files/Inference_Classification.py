import os
import cv2
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from Model_Scratch import MODEL as MODEL_SCRATCH
from Model_Transfer_Learning import MODEL as MODEL_TRANSFER

def load_label_map(textFile):
    return np.loadtxt(textFile, str, delimiter='\t')
    
def cv_image_read(image_path):
    print(image_path)
    return cv2.imread(image_path)

def print_result(inference_result, class_map):
    class_text = class_map[np.argmax(inference_result)]
    print(inference_result)
    print(class_text)

def inference_image(opencv_image, transform_info, model, DEVICE):
    image = Image.fromarray(opencv_image)
    image_tensor = transform_info(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)
    result = model(image_tensor)
    return result


def inference_main(image_path, _is_transfer = False):
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    img_width, img_height = 224, 224
    transform_info = transforms.Compose([
                transforms.Resize(size=(img_width, img_height))
                , transforms.ToTensor()
                    ])
    class_map = load_label_map('label_map.txt')
    num_classes = len(class_map)
    
    model = None
    model_str = None
    if not _is_transfer:
        model = MODEL_SCRATCH(num_classes).to(DEVICE)
        model_str = "PyTorch_Classification_Model_Scratch"
    else:
        model = MODEL_TRANSFER(num_classes).to(DEVICE)
        model_str = "PyTorch_Classification_Model_Trnasfer"
    model_str += ".pt" 
    
    model.load_state_dict(torch.load(model_str))
    model.eval()

    opencv_image = cv_image_read(image_path)
    inference_result = inference_image(opencv_image, transform_info, model, DEVICE)
    inference_result = inference_result.cpu().detach().numpy()
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
