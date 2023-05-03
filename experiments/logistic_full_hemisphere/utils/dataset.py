import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from tqdm import tqdm

import numpy as np

from matplotlib import pyplot as plt
# custom dataset used to load pairs for VAE training
class ReferenceDataset(Dataset):
    def __init__(self, directory, side):
        self.directory = directory

        self.transform = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x224 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])

        print("Loading dataset sample names...")

        train_img_dir = self.directory + "/training_split/training_images"
        test_img_dir = self.directory + "/test_split/test_images"

        # Create lists will all training and test image file names, sorted
        self.train_img_list = os.listdir(train_img_dir)
        self.train_img_list.sort()
        self.test_img_list = os.listdir(test_img_dir)
        self.test_img_list.sort()
        print('Training images: ' + str(len(self.train_img_list)))
        print('Test images: ' + str(len(self.test_img_list)))


        fmri_dir = self.directory + "/training_split/training_fmri"
        
        self.fmri = np.load(os.path.join(fmri_dir, side + '_training_fmri.npy'))
        self.fr_fmri = self.fmri

        self.mean = np.mean(self.fmri, axis=0)
        self.maxi = np.max(self.fmri, axis=0)
        self.mini = np.min(self.fmri, axis=0)

        self.fmri = (self.fmri - self.mini)/(self.maxi - self.mini)
        self.fmri = 2*(self.fmri - self.mean)
        self.fmri = self.fmri - 1


    def __len__(self):
        return len(self.train_img_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.transform(Image.open(self.directory + "/training_split/training_images/" + self.train_img_list[idx]))
        label = self.fmri[idx]

        return (image, label)
    
    def getStats(self):
        return (self.mean, self.maxi, self.mini)
    
    def getVertexCount(self):
        return self.vertex_count

    def getFrFmri(self, idx):
        return self.fr_fmri[idx]
    
    def getLabels(self, idx):
        return self.fmri[idx]