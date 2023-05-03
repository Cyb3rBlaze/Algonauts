from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights

from torchvision.ops import FeaturePyramidNetwork

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from tqdm import tqdm

import numpy as np

import os

from utils.dataset_test import Dataset
from utils.dataset import ReferenceDataset
from utils.model import RegressionHead

def test(subject, side):
    hemisphere = "lh" if side == "left" else "rh"

    dataset = Dataset("../../data/" + subject)
    reference_dataset = ReferenceDataset("../../data/" + subject, side=hemisphere)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    stats = reference_dataset.getStats()
    vertex_count = reference_dataset.getVertexCount()
    # loading pretrained model
    device = torch.device("cuda")

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    feature_extractor = create_feature_extractor(model, 
            return_nodes=["layer1.1.conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]).to(device)
    fpn = FeaturePyramidNetwork([64, 128, 256, 512], 256).to(device)
    feature_extractor.eval()


    # instantiating trainable head
    head = RegressionHead(vertex_count).to(device)
    head.load_state_dict(torch.load("saved_models/5_epochs_aux_" + subject + "_" + side))
    head.eval()

    # generating test outputs
    with torch.no_grad():
        inputs = torch.stack(tuple(dataset), dim=0).to(device)

        # feature extractor backbone
        print(f"extracting features for {subject}, side {side}")
        outputs = feature_extractor(inputs)
        outputs = fpn(outputs)

        # trainable head
        print(f"predicting fmri for {subject}, side {side}")
        outputs = head(outputs)
        mean = stats[0]
        maxi = stats[1]
        mini = stats[2]

        
        print(f"saving outputs for {subject}, side {side}")
        outputs = outputs.to("cpu").numpy().astype(np.float32)

        o_maxi = outputs.max(axis=0)
        o_mini = outputs.min(axis=0)

        outputs = (outputs - o_mini)/(o_maxi - o_mini)
        outputs = outputs*(maxi-mini)
        outputs = outputs + mini
        outputs = outputs/2
        

        np.save('algonauts_2023_challenge_submission/' + subject + '/' + hemisphere + '_pred_test.npy', outputs)