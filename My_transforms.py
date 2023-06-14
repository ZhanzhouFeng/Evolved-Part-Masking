import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torchvision.transforms.functional as F
import cv2
import numpy as np


class RandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img,target):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation),F.resized_crop(target, i, j, h, w, self.size, transforms.InterpolationMode.NEAREST)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img,target):
        if torch.rand(1) < self.p:
            return F.hflip(img),F.hflip(target)
        return img,target

class ToTensor(transforms.ToTensor):
    def __call__(self, img,target):
        target=cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2BGR)
        return F.to_tensor(img),torch.tensor(target,dtype=torch.float32)

class Normalize(transforms.Normalize):
    def forward(self, img,target):
        return F.normalize(img, self.mean, self.std, self.inplace),target


class Compose(transforms.Compose):
    def __call__(self, img,target):
        for t in self.transforms:
            img,target = t(img,target)
        return img,target

