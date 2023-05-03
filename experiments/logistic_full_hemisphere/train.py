from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from tqdm import tqdm

import numpy as np

import os

from utils.dataset_both import Dataset
from utils.model import RegressionHead

import wandb


# loading dataset + creating train test split for verifying performance
def train(subject, hemisphere):
    config={
        "subject": subject,
        "hemisphere": hemisphere,
        "learning_rate": 1e-3,
        "architecture": "logistic_full_hemisphere",
        "dataset": "Algonauts",
        "epochs": 5,
        "batch_size": 32,
        "l2": 0.00002
    }
    run = wandb.init(project="resnet_feature_regression", config=config, entity="algonauts")

    dataset = Dataset(f"../../data/subj0{subject}")

    train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=run.config.batch_size, shuffle=True)

    # loading pretrained model
    device = torch.device("cuda")

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    layer_names = []

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer_names += [name]

    feature_extractor = create_feature_extractor(model, 
            return_nodes=["layer1.1.conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]).to(device)
    fpn = FeaturePyramidNetwork([64, 128, 256, 512], 256).to(device)

    feature_extractor.eval()

    fmri = np.load(os.path.join(f"../../data/subj0{subject}/training_split/training_fmri/", 'lh_training_fmri.npy' if hemisphere=="left" else 'rh_training_fmri.npy' ))
    # initiating trainable head
    head = RegressionHead(fmri.shape[1]).to(device)
    print("num_vertices", fmri.shape[1])

    # used to make spread more closely match target distribution
    def squared_spread_loss():
        def loss(output, target):
            output_std = torch.std(output)
            target_std = torch.std(target)
            return (output_std - target_std) ** 2
        return loss

    mse_weight = 0.6
    spread_weight = 0.4

    criterion = nn.MSELoss() # loss
    auxiliary_spread_loss = squared_spread_loss() # auxiliary loss
    optimizer = torch.optim.Adam(head.parameters(), lr=run.config.learning_rate, weight_decay=run.config.l2) # optimizer

    train_loss = 0
    mse_train_loss = 0
    val_loss = 0
    mse_val_loss = 0
    count = 0

    all_train_loss_vals = []
    all_val_loss_vals = []

    for epoch in range(run.config.epochs):
        # setting to train mode for gradient calculations
        head.train()

        for i, (inputs, left_targets, right_targets) in tqdm(enumerate(train_loader), total=int(len(train_set)/run.config.batch_size)+1):
            inputs = inputs.to(device)
            targets = left_targets.to(device) if run.config.hemisphere == "left" else right_targets.to(device)

            optimizer.zero_grad()

            # feature extractor backbone
            outputs = feature_extractor(inputs)
            outputs = fpn(outputs)

            # trainable head
            outputs = head(outputs)

            mse_loss = (mse_weight * criterion(outputs, targets))
            loss = mse_loss + (spread_weight * auxiliary_spread_loss(outputs, targets))
            loss.backward()

            del inputs
            del targets
            del outputs

            count += 1

            mse_train_loss = mse_train_loss + mse_loss.item()
            train_loss = train_loss + loss.item()

            optimizer.step()

            torch.cuda.empty_cache() # frees up memory for val

        all_train_loss_vals += [(mse_train_loss * 10 / 6)/count]
        
        wandb.log({"Train loss": train_loss/count, "Raw MSE train loss": (mse_train_loss * 10 / 6)/count})
        
        count = 0

        head.eval()
        
        with torch.no_grad():
            for i, (inputs, left_targets, right_targets) in tqdm(enumerate(val_loader), total=int(len(val_set)/run.config.batch_size)+1):
                inputs = inputs.to(device)
                #targets = targets.to(device)[:, np.where(challenge_roi)[0]] # selecting proper vertices based on ROI
                targets = left_targets.to(device) if run.config.hemisphere == "left" else right_targets.to(device)
                
                # feature extractor backbone
                outputs = feature_extractor(inputs)
                outputs = fpn(outputs)

                # trainable head
                outputs = head(outputs)

                mse_loss = (mse_weight * criterion(outputs, targets))
                loss = mse_loss + (spread_weight * auxiliary_spread_loss(outputs, targets))

                del inputs
                del targets
                del outputs

                count += 1

                mse_val_loss = mse_val_loss + mse_loss.item()
                val_loss = val_loss + loss.item()

                torch.cuda.empty_cache() # frees up memory for val

        all_val_loss_vals += [(mse_val_loss * 10 / 6)/count]
        
        wandb.log({"Val loss": val_loss/count, "Raw MSE val loss": (mse_val_loss * 10 / 6)/count})

        train_loss = 0
        mse_train_loss = 0
        val_loss = 0
        mse_val_loss = 0
        
        count = 0

    torch.save(head.state_dict(), f"{run.config.epochs}_epochs_aux_subj0{run.config.subject}_{run.config.hemisphere}")
    wandb.save(f"{run.config.epochs}_epochs_aux_subj0{run.config.subject}_{run.config.hemisphere}")
    wandb.finish()