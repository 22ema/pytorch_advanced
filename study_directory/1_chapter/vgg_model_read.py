import numpy as np
import json
from PIL import Image
import matplotlib.pyplt as plt

import torch
import torchvision
from torchvision import models, transforms

if __name__ == "__main__":
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.eval()