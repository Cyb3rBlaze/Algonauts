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
    def __init__(self):
        self.directories = [f"../../data/subj0{i}" for i in range(1,6)]

        self.transform = transforms.Compose([
            transforms.Resize((224,224)), # resize the images to 224x224 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        ])

        print("Loading dataset sample names...")

        train_img_dirs = [f"{directory}/training_split/training_images" for directory in self.directories]
        test_img_dirs = [f"{directory}/test_split/test_images" for directory in self.directories]

        # Create lists will all training and test image file names, sorted
        self.train_img_list = []
        for train_img_dir in train_img_dirs:
            self.train_img_list.append(os.listdir(train_img_dir))
        
        self.test_img_list = []
        for test_img_dir in test_img_dirs:
            self.test_img_list.append(os.listdir(test_img_dir))
        

        fmri_dirs = [f"{directory}/training_split/training_fmri" for directory in self.directories]
        
        self.lh_fmris = []
        for fmri_dir in fmri_dirs:
            self.lh_fmris.append(np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy')))

        for i in range(len(self.lh_fmris)):
            lh_fmri = self.lh_fmris[i]
            mean = np.mean(lh_fmri, axis=0)
            maxi = np.max(lh_fmri, axis=0)
            mini = np.min(lh_fmri, axis=0)
            self.lh_fmris[i] = 2*(lh_fmri - mini)/(maxi - mini) - 1

            print('\nLH training fMRI data shape:')
            print(self.lh_fmris[i].shape)
            print('(Training stimulus images × LH vertices)')
            plt.hist(self.lh_fmris[i].flatten(), bins=100)

        self.starting_points = [0] + [len(train_img_list) for train_img_list in self.train_img_list]
        self.starting_points = np.cumsum(self.starting_points)

    def __len__(self):
        length = 0
        for train_img_list in self.train_img_list:
            length += len(train_img_list)
        return length


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        list_of_starting_points = self.starting_points - idx
        subject = np.argmax(list_of_starting_points>0)-1

        idx = idx - self.starting_points[subject]

        image = self.transform(Image.open(self.directories[subject] + "/training_split/training_images/" + self.train_img_list[subject][idx]))
        left_label = self.lh_fmris[subject][idx]

        return (image, left_label)