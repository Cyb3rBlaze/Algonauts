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
        
        self.lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))

        print('\nLH training fMRI data shape:')
        print(self.lh_fmri.shape)
        print('(Training stimulus images × LH vertices)')
        plt.hist(self.lh_fmri.flatten(), bins=200)


    def __len__(self):
        return len(self.train_img_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.transform(Image.open(self.directory + "/training_split/training_images/" + self.train_img_list[idx]))
        left_label = self.lh_fmri[idx]

        return (image, left_label)