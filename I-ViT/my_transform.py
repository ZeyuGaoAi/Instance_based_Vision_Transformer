import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None, weight=None):
        x=0
        for t in self.transforms:
            image = t(image)
#             image, target, weight = t(image, target, weight)
        return image
#         return image, target, weight

class Compose2(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, weight):
        for t in self.transforms:
            image, target, weight = t(image, target, weight)
        return image, target, weight

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, weight):
        size = self.size
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        weight = F.resize(weight, size, interpolation=Image.NEAREST)
        return image, target, weight

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target, weight):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        weight = F.resize(weight, size, interpolation=Image.NEAREST)
        return image, target, weight


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None, weight=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
#             target = F.hflip(target)
#             weight = F.hflip(weight)
        return image
#         return image, target, weight


class RandomCrop(object):
    def __init__(self, base_size, size):
        self.base_size = base_size
        self.size = size

    def __call__(self, image, target, weight):
        image = pad_if_smaller(image, self.base_size)
        target = pad_if_smaller(target, self.base_size, fill=0)
        weight = pad_if_smaller(weight, self.base_size, fill=0)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        weight = F.crop(weight, *crop_params)
        return image, target, weight
    
class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, target=None, weight=None):
        angel_params = T.RandomRotation.get_params(self.angle)
        image = F.rotate(image, angel_params)
#         target = F.rotate(target, angel_params)
#         weight = F.rotate(weight, angel_params)
        return image
#         return image, target, weight

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, weight):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        weight = F.center_crop(weight, self.size)
        return image, target, weight


class ToTensor(object):
    def __call__(self, image, target, weight):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
        weight = torch.as_tensor(np.asarray(weight), dtype=torch.int64)
        #return image
        return image, target, weight
    
class ToTensor2(object):
    def __call__(self, image, target=None, weight=None):
        image = F.to_tensor(image)
#         target = torch.as_tensor(np.asarray(target), dtype=torch.int64)
#         weight = torch.as_tensor(np.asarray(weight), dtype=torch.int64)
        return image
#         return image, target, weight

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None, weight=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image
#         return image, target, weight