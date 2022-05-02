from inference import Inference
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import time
import os
import numpy as np
class TrainModel():
    def __init__(self, model):
        self.model = model
        self.use_pretrained = True
        self.batch_size = 10
        self.init_lr = 0.0001
        self.num_epochs = 10
        self.use_gpu = False
        self.dataset_size = None
        self.valid_size = 0.2
        self.train_valid_dataloader = None
        self.test_dataloader = None

    def select_model(self):
        if self.model == "resnet18":                                            ## change grayscale input channel
            model_ft = models.resnet18(pretrained=self.use_pretrained)
            model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            return model_ft
        elif self.model == "inception_v3":
            model_ft = models.inception_v3(pretrained=self.use_pretrained)
            model_ft.Conv2d_1a_3x3 = models.inception.BasicConv2d(1, 32, kernel_size=3, stride=2)
            model_ft.transform_input = False
            model_ft.aux_logits = False
            return model_ft
        elif self.model == "densenet":
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            model_ft.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                    padding=3, bias=False)
            return model_ft
        else:
            print("input the wrong model")

    def train(self):
        self.train_valid_dataloader = self.make_train_valid_dataset()
        self.test_dataloader = self.make_test_dataset()
        net = self.select_model()
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(net.parameters(), lr=float(self.init_lr), momentum=0.9)
        lr_schedule = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        self.training_model(net, criterion, lr_schedule, optimizer_ft)

    def training_model(self,net, criterion, lr_schedule, optimizer):
        self.use_gpu = torch.cuda.is_available()
        since = time.time()
        best_model_wts = net.state_dict()
        best_acc = 0.0
        if self.use_gpu:
            net = net.cuda()
        for epoch in range(self.num_epochs):
            print("Epoch{}/{}".format(epoch, self.num_epochs-1))
            print('-'*10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train(True)
                else:
                    net.train(False)
                running_loss = 0.0
                running_corrects = 0
                print(len(self.train_valid_dataloader[phase]))
                for inputs, labels in tqdm(self.train_valid_dataloader[phase]):
                    if self.use_gpu :
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else :
                        inputs, labels = Variable(inputs), Variable(labels)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        lr_schedule.step()
                    running_loss += loss.data
                    running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss/len(self.train_valid_dataloader[phase])
            epoch_acc = float(running_corrects)/len(self.train_valid_dataloader[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = net.state_dict()
            inference_model = Inference(net, self.test_dataloader, self.use_gpu)
            inference_model.inference()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            float(time_elapsed) // 60.0, float(time_elapsed) % 60.0))
        print('Best val Acc : {:4f}'.format(best_acc))
        dir_name = time.strftime('%c', time.localtime(time.time()))
        os.mkdir("./logs/"+dir_name)
        net.load_state_dict(best_model_wts)
        torch.save(net.state_dict(), 'logs/'+dir_name+"/"+self.model+'_'+str(self.num_epochs)+'_'+'.pth')




    def make_train_valid_dataset(self):
        if self.model == "inception_v3":
            train_data_transforms = transforms.Compose([
                    transforms.Resize(299),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            valid_data_transforms = transforms.Compose([
                    transforms.Resize(299),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

        else:
            train_data_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            valid_data_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
        mnist_train = datasets.MNIST(root = './dataset/',
                                     train=True,
                                     transform = train_data_transforms,
                                     download = True)
        mnist_validation = datasets.MNIST(root = './dataset/',
                                          train=True,
                                          transform = valid_data_transforms,
                                          download=True)
        num_train = len(mnist_train)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size*num_train))
        np.random.seed(3)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_data_loader = torch.utils.data.DataLoader(dataset = mnist_train,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler)
        val_data_loader = torch.utils.data.DataLoader(dataset = mnist_validation,
                                                      batch_size=self.batch_size,
                                                      sampler=valid_sampler)
        dataloader_dict = {"train" : train_data_loader, "val" : val_data_loader}
        return dataloader_dict

    def make_test_dataset(self):
        if self.model == "inception_v3":
            test_data_transforms = transforms.Compose([
                    transforms.Resize(299),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

        else:
            test_data_transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
        test_dataset = datasets.MNIST(root="./dataset",
                                      train= False,
                                      download=True,
                                      transform=test_data_transforms)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=10)
        return test_data_loader

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='train Convolution network on MNIST dataset')
    parser.add_argument("--model", required=True, help="be trained model(resnet18, vgg16, inception_v3, densenet")
    args = parser.parse_args()
    print("Model:", args.model)
    do_train = TrainModel(args.model)
    do_train.train()