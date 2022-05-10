import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models
from utils import data_transform

class ILSVRCPredictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]
        return predicted_label_name

if __name__ == "__main__":
    image_path = "./dataset/1.jpg"
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.299, 0.224, 0.225)
    base_transform = data_transform.BaseTransform(resize, mean, std)
    image_transform = base_transform(img)
    image_transform = image_transform.numpy().transpose((1,2,0))
    image_transform = np.clip(image_transform, 0, 1)
    plt.imshow(image_transform)
    plt.show()
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.eval()
