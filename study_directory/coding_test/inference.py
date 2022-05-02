import torch
from torchvision import models, transforms, datasets
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
class Inference():
    def __init__(self, model, dataloader, use_gpu):
        self.model = model
        self.dataloader = dataloader
        self.use_gpu = use_gpu

    def inference(self):
        net = self.model.eval()
        for inputs, labels in tqdm(self.dataloader['test']):
            if self.use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            softmax = nn.Softmax(dim=1)
            output = softmax(output)
            value, preds = torch.max(output.data, 1)
            print(labels, preds)

