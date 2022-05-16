import os
import glob

import torch.cuda

from utils.MakeDataset import HymenopteraDataset
from utils.data_transform import ImageTransform
from torch.utils.data import dataloader
from torchvision import models
from torch import nn
import torch.optim as optim

def make_image_list(image_path, phase="train"):
    image_path = os.path.join(image_path, phase, "*/*.jpg")
    image_list = list()
    for image in glob.glob(image_path):
        image_list.append(image)
    return image_list

def train_model(net, dataloader_dict, criterion, optimizer, epoch_num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 장치:", device)
    net.to(device)
    # 네트워크가 어느 정도
    for epoch in range(epoch_num):
        print('Epoch {}/{}'.format(epoch+1, epoch_num))
        print('---------------------------------------')
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            elif phase == 'val':
                net.eval()
            epoch_loss = 0
            epoch_corrects = 0.0
            if epoch == 0 and phase == 'train':
                continue
            for inputs, labels in dataloader_dict[phase]:
                inputs.to(device)
                labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = net(inputs)
                    loss = criterion(output, labels)
                    _, preds = torch.max(output, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # 손실 합계 갱신 (batch size만큼?)
                    epoch_loss += loss.item() * inputs.size(0)
                    # 정답 수의 합계 갱신
                    epoch_corrects += torch.sum(preds == labels.data)
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))



if __name__ == "__main__":
    number_epoch = 2
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.299, 0.224, 0.225)
    batch_size = 32
    use_pretrained_model = True
    criterion = nn.CrossEntropyLoss()
    # 파인튜닝으로 학습할 파라미터를 params_to_update 변수의 1~3에 저장
    params_to_update_1 = list()
    params_to_update_2 = list()
    params_to_update_3 = list()

    # 학습시킬 층의 파라미터명 지정
    update_param_names_1 = ["features"]
    update_param_names_2 = ["classifier.0.weight",
                            "classifier.0.bias", "classifier.3.weight",
                            "classifier.3.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

    optimizer = optim.SGD([
        {"params": params_to_update_1, "lr": 1e-4},
        {"params": params_to_update_2, "lr": 5e-4},
        {"params": params_to_update_3, "lr": 1e-3}
    ], momentum= 0.9)

    # make data loader
    image_path = "./dataset/hymenoptera_data"
    train_image_list = make_image_list(image_path, "train")
    val_image_list = make_image_list(image_path, "val")
    image_trans = ImageTransform(size, mean, std)
    train_dataset = HymenopteraDataset(train_image_list, transform=image_trans, phase="train")
    val_dataset = HymenopteraDataset(val_image_list, transform=image_trans, phase="val")
    train_dataloader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

    # load network
    net = models.vgg16(pretrained=use_pretrained_model)

    # 파라미터를 각 리스트에 저장
    for name, param in net.named_parameters():
        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)
            print("params_to_update_1에 저장: ", name)
        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)
            print("params_to_update_2에 저장: ", name)
        elif name in update_param_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)
            print("params_to_update_3에 저장: ", name)
        else:
            param.requires_grad = False
            print("경사 계산 없음: ", name)
    train_model(net, dataloader_dict, criterion, optimizer, number_epoch)
