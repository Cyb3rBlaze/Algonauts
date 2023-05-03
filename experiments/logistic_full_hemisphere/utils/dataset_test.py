import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from tqdm import tqdm

import numpy as np

from matplotlib import pyplot as plt
# custom dataset used to load pairs for VAE training
class Dataset(Dataset):
    def __init__(self, directory):
        self.directory = directory

        self.transform = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x224 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])

        print("Loading dataset sample names...")

        test_img_dir = self.directory + "/test_split/test_images"

        # Create lists will all training and test image file names, sorted
        self.test_img_list = os.listdir(test_img_dir)
        self.test_img_list.sort()
        print('Test images: ' + str(len(self.test_img_list)))


    def __len__(self):
        return len(self.test_img_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.transform(Image.open(self.directory + "/test_split/test_images/" + self.test_img_list[idx]))

        return (image)