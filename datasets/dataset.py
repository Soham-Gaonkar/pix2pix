from PIL import Image
from os import listdir
import torchvision.transforms as transforms
import numpy as np

class Dataset():

    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.files = [f for f in listdir(data_path) if f.endswith(".jpg")]
        self.len = len(self.files)

        self.transform_list = []
        self.transform_list += [transforms.ToTensor()]
        self.transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        transform_list = self.transform_list.copy()
        if np.random.rand() > 0.5: 
            transform_list = [transforms.RandomHorizontalFlip(p=1.0)] + transform_list
        transform = transforms.Compose(transform_list)

        AB = Image.open(self.data_path + "/" + self.files[index])
        w, h = AB.size
        B = AB.crop((0, 0, w // 2, h))  # Output
        A = AB.crop((w // 2, 0, w, h))  # Input
        return transform(A), transform(B)
