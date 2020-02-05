# -*- coding: utf-8 -*-
"""PyTorch_Image_Classification_Training_From_Scratch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1h0AAIYAx_vCO0Ed1X2oTNqPDRfqgPh6z
"""

!wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
!unzip -qq cats_and_dogs_filtered.zip

import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class PyTorchCustomDataset(Dataset):
    def __init__(self
                 , root_dir = "/content/cats_and_dogs_filtered/train"
                 , transform = None):
        self.image_abs_path = root_dir
        self.transform = transform
        self.label_list = os.listdir(self.image_abs_path)
        self.label_list.sort()
        self.x_list = []
        self.y_list = []
        for label_index, label_str in enumerate(self.label_list):
            img_path = os.path.join(self.image_abs_path, label_str)
            img_list = os.listdir(img_path)
            for img in img_list:
                self.x_list.append(os.path.join(img_path, img))
                self.y_list.append(label_index)
        pass

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        image = Image.open(self.x_list[idx])
        if image.mode is not "RGB":
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.y_list[idx]

    def __save_label_map__(self, dst_text_path = "label_map.txt"):
        label_list = self.label_list
        f = open(dst_text_path, 'w')
        for i in range(len(label_list)):
            f.write(label_list[i]+'\n')
        f.close()
        pass

    def __num_classes__(self):
        return len(self.label_list)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MODEL(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1)
            , nn.BatchNorm2d(32)
            , nn.ReLU()
            , nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1)
            , nn.BatchNorm2d(64)
            , nn.ReLU()
            , nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1)
            , nn.BatchNorm2d(128)
            , nn.ReLU()
            , nn.AdaptiveAvgPool2d(1)
            , nn.Flatten()
            , nn.Linear(128, 64)
            , nn.ReLU()
            , nn.Dropout()
            , nn.Linear(64, num_classes)
            , nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier(x)

import torch
import torch.optim as optim

def main():
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    img_width, img_height = 224, 224
    EPOCHS     = 50
    BATCH_SIZE = 16
    transform_train = transforms.Compose([
                transforms.Resize(size=(img_width, img_height))
                , transforms.RandomRotation(degrees=15)
                , transforms.ToTensor()
                ])
    transform_test = transforms.Compose([
                transforms.Resize(size=(img_width, img_height))
                , transforms.ToTensor()
                ])

    TrainDataset = PyTorchCustomDataset
    TestDataset = PyTorchCustomDataset

    train_data = TrainDataset(root_dir = "/content/cats_and_dogs_filtered/train"
                    , transform = transform_train)
    test_data = TestDataset(root_dir = "/content/cats_and_dogs_filtered/validation"
                    , transform = transform_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_data
        , batch_size=BATCH_SIZE
        , shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data
        , batch_size=BATCH_SIZE
        , shuffle=True
    )
    
    train_data.__save_label_map__()
    num_classes = train_data.__num_classes__()

    model = MODEL(num_classes).to(DEVICE)
    model_str = "PyTorch_Classification_Model"
    model_str += ".pt" 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, target in (train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in (test_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)

                # 배치 오차를 합산
                test_loss += F.cross_entropy(output, target,
                                            reduction='sum').item()

                # 가장 높은 값을 가진 인덱스가 바로 예측값
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)

        print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                epoch, test_loss, test_accuracy))

        if acc < test_accuracy:
            acc = test_accuracy
            torch.save(model.state_dict(), model_str)
            print("model saved!")

main()