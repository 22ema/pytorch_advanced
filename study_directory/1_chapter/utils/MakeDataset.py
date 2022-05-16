
from PIL import Image

import torch.utils.data as data

class HymenopteraDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        '''
        file_list 길이를 반환
        :return: len(self.file_lit)
        '''
        return len(self.file_list)

    def __getitem__(self, index): ## __getitem__은 클래스의 인덱스에 접근할 때 자동으로 호출되는 메서드이다.
        '''
        전처리한 화상의 텐서 형식의 데이터와 라벨 return
        :param index:
        :return:
        '''

        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transform = self.transform(img, self.phase)

        ## label을 파일 이름에서 추출
        if self.phase == "train" :
            label = img_path[33:37]
        elif self.phase == "val":
            label = img_path[31:35]

        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1
        return img_transform, label

