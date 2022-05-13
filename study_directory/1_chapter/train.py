import glob
import os
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.data_transform import ImageTransform
from utils.MakeDataset import HymenopteraDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

def make_data_path_list(file_path, phase="train"):
    target_path = os.path.join(file_path, phase, '*/*.jpg')
    path_list = list()
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('---------------------------------------')
        for phase in ['train', 'val']:
            if phase == 'train':    #학습 모드
                net.train()
            elif phase == 'val':    #검증 모드
                net.eval()
            epoch_loss = 0.0        # epoch loss 합
            epoch_corrects = 0      # epoch 정답 수
            # 학습 하지 않을 시 성능 검사를 위해 epoch=0의 훈련 생략
            if (epoch == 0) == (phase == 'train'):
                continue
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # 옵티마이저 초기화
                optimizer.zero_grad()
                # 순전파 계산
                with torch.set_grad_enabled(phase =="train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1) #라벨 예측 (tensor 내 최대 값 반환)
                    # 훈련 시에는 오차 역전파
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    #손실 합계 갱신 (batch size만큼?)
                    epoch_loss += loss.item()*inputs.size(0)
                    #정답 수의 합계 갱신
                    epoch_corrects += torch.sum(preds==labels.data)
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double()/len(dataloaders_dict[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))



if __name__ == "__main__":
    img_file_path = "./dataset/hymenoptera_data"
    train_img_list = make_data_path_list(img_file_path, "train")
    val_img_list = make_data_path_list(img_file_path, "val")
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.299, 0.224, 0.225)
    batch_size = 32
    use_pretrained = True
    train_dataset = HymenopteraDataset(file_list=train_img_list, transform=ImageTransform(size, mean, std), phase= "train")
    val_dataset = HymenopteraDataset(file_list=val_img_list, transform=ImageTransform(size, mean, std), phase= "val")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    dataloader_dict = {"train": train_dataloader, "val": val_dataloader}
    batch_iterator = iter(dataloader_dict["train"])
    inputs, labels = next(batch_iterator)

    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    net.train()
    print("train mode ON")

    criterion = nn.CrossEntropyLoss()

    # 전이학습에서 학습 시킬 파라미터를 params_to_update 변수에 저장
    params_to_update = list()
    # 학습시킬 파라미터명
    update_params_names = ["classifier.6.weight", "classifier.6.bias"]
    #학습시킬 파라미터 외에는 경사 계산하지 않고 변하지 않도록 설정
    for name, params in net.named_parameters():
        if name in update_params_names:
            params.required_grad = True
            params_to_update.append(params)
            print(name)
        else:
            params.required_grad = False
    #최적화 기법 설정
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
    train_model(net, dataloader_dict, criterion, optimizer, 2)
