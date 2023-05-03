from torchvision.models.feature_extraction import create_feature_extractor
# from torchvision.models import resnet50, ResNet50_Weights

from torchvision.ops import FeaturePyramidNetwork

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import gc

from tqdm import tqdm

import numpy as np

import os

import matplotlib.pyplot as plt

from utils.dataset import Dataset

from scipy.stats import pearsonr as corr

from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression

batch_size = 100

for i in range(1, 9):
    for j in range(2):
        if j == 0:
            right = False
        else:
            right = True
        data = Dataset(f'../../data/subj0{i}')
        test_data = Dataset(f'../../data/subj0{i}', test=True)

        #train_set, val_set = torch.utils.data.random_split(data, [0.8, 0.2])

        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        #val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
        alexnet.to(device) # send the alexnet to the chosen device ('cpu' or 'cuda')
        alexnet.eval() # set the alexnet to evaluation mode, since you are not training it

        model_layer = "features.2" #@param ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"] {allow-input: true}
        feature_extractor = create_feature_extractor(alexnet, return_nodes=[model_layer])
        feature_extractor.to(device)
        feature_extractor.eval()

        def fit_pca(feature_extractor, dataloader):
            # Define PCA parameters
            pca = IncrementalPCA(n_components=100, batch_size=batch_size)

            # Fit PCA to batch
            for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                if _ == len(dataloader)-1:
                    break
                # Extract features
                ft = feature_extractor(d[0].to(device))
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
                # Fit PCA to batch
                pca.partial_fit(ft.detach().cpu().numpy())

            return pca
        pca = fit_pca(feature_extractor, train_loader)

        def extract_features(feature_extractor, dataloader, pca, right=False, test=False):
            fmri = []
            features = []
            for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Extract features
                if test == False:
                    ft = feature_extractor(d[0].to(device))
                    if right == False:
                        fmri += [d[1].cpu().detach().numpy()]
                    else:
                        fmri += [d[2].cpu().detach().numpy()]
                else:
                    ft = feature_extractor(d.to(device))
                # Flatten the features
                ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
                # Apply PCA transform
                ft = pca.transform(ft.cpu().detach().numpy())
                features += [ft]
            if test == False:
                return (np.vstack(features), np.vstack(fmri))
            return np.vstack(features)
        
        features_train, labels_train = extract_features(feature_extractor, train_loader, pca, right=right)
        features_test = extract_features(feature_extractor, test_loader, pca, right=right, test=True)

        reg = LinearRegression().fit(features_train, labels_train)
        fmri_test_pred = reg.predict(features_test)

        fmri_test_pred = fmri_test_pred.astype(np.float32)
        np.save(os.path.join(f"algonauts_2023_challenge_submission_alexnet/subj0{i}/", 'rh_pred_test.npy' if right else 'lh_pred_test.npy'), fmri_test_pred)

        torch.cuda.empty_cache()
        gc.collect()