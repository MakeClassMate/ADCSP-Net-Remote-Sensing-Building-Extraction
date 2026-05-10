import os
import numpy as np
import torch
import random
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

class ImageLabelDataset(Dataset):
    def __init__(self, image_folder, label_folder,transform=None):
        self.image_folder = image_folder

        self.label_folder = label_folder
        self.transform = transform

        self.image_filenames = os.listdir(image_folder)

        self.label_filenames = os.listdir(label_folder)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]

        label_name = self.label_filenames[idx]

        image_path = os.path.join(self.image_folder, image_name)

        label_path = os.path.join(self.label_folder, label_name)

        image = cv2.imread(image_path)

        label = cv2.imread(label_path, 0)/255.
        # image = cv2.resize(image, (5, 256))
        #
        # label = cv2.resize(label, (256, 256))

        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(label).long()

        return image, label

def worker_init_fn(worker_id):
    # 每个 worker 有不同但可控的随机种子
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

    
def prepare_dataloader(image_folder, label_folder, batch_size, shuffle,drop_last=False, num_workers=4):
    transform = ToTensor()  # 可以根据需要添加其他数据增强操作

    dataset = ImageLabelDataset(image_folder, label_folder, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last, num_workers=num_workers, worker_init_fn=worker_init_fn,pin_memory=True)

    return loader

# import os
# import cv2
# import torch
# from torch.utils.data import Dataset, DataLoader
# import albumentations as albu
# from albumentations.pytorch import ToTensorV2


# class ImageLabelDataset(Dataset):
#     def __init__(self, image_folder, label_folder, transform=None):
#         self.image_folder = image_folder
#         self.label_folder = label_folder
#         self.transform = transform

#         self.image_filenames = os.listdir(image_folder)
#         self.label_filenames = os.listdir(label_folder)

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, idx):
#         image_name = self.image_filenames[idx]
#         label_name = self.label_filenames[idx]

#         image_path = os.path.join(self.image_folder, image_name)
#         label_path = os.path.join(self.label_folder, label_name)

#         # 读取图像和标签
#         image = cv2.imread(image_path)
#         label = cv2.imread(label_path, 0)  # 读取为单通道灰度图

#         # 数据增强
#         if self.transform:
#             augmented = self.transform(image=image, mask=label)  # 同时对图像和标签进行增强
#             image = augmented['image']
#             label = augmented['mask']

#         # 转换标签为Long类型（用于分类任务）
#         label = label.long()

#         return image, label


# # 训练集的数据增强流水线
# def get_train_transform():
#     return albu.Compose([
#         albu.Flip(p=0.6),  # 随机水平翻转，60%的概率
#         albu.RandomRotate90(p=0.6),  # 随机旋转90度，60%的概率
#         albu.ShiftScaleRotate(scale_limit=0.5, shift_limit=0.5, rotate_limit=180, p=0.6),  # 随机缩放、平移和旋转，60%的概率
#         ToTensorV2()  # 将图像转换为Tensor
#     ])


# # 验证集的数据增强流水线
# def get_validation_augmentation():
#     """Add paddings to make image shape divisible by 32"""
#     return albu.Compose([
#         albu.Flip(p=0.6),  # 随机水平翻转，50%的概率
#         albu.RandomRotate90(p=0.6),  # 随机旋转90度，30%的概率
#         albu.ShiftScaleRotate(scale_limit=0.2, shift_limit=0.2, rotate_limit=0, p=0.6),  # 小幅度的平移、缩放和旋转，30%的概率
#         ToTensorV2()  # 将图像转换为Tensor
#     ])


# def prepare_dataloader(image_folder, label_folder, batch_size, shuffle, num_workers=4, mode='train'):
#     # 根据mode来选择使用不同的transform
#     if mode == 'train':
#         transform = get_train_transform()  # 训练集使用激烈的增强
#     elif mode == 'val':
#         transform = get_validation_augmentation()  # 验证集使用轻度的增强

#     dataset = ImageLabelDataset(image_folder, label_folder, transform=transform)

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

#     return loader
