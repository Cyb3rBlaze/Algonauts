from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights

import torch
import torch.nn as nn

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

from utils.dataset import Dataset

import os
import gc


batch_size = 32
EPOCHS = 100


class RegressionHead(torch.nn.Module):
    def __init__(self, output_size):
        super(RegressionHead, self).__init__()

        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(1024, 512, 3)
        self.pool = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(18432, 6000)
        self.linear2 = nn.Linear(6000, output_size)

    def forward(self, x):
        output = self.conv1(x)
        output = self.pool(output)

        output = self.flatten(output)

        output = self.linear1(output)

        return self.linear2(output)
    

for i in range(5, 9):
    for j in range(2):
        if j == 0:
            right = False
        else:
            right = True
        

        if i == 6:
            if j == 0:
                output_size = 18978
            else:
                output_size = 20220
        elif i == 8:
            if j == 0:
                output_size = 18981
            else:
                output_size = 20530
        else:
            if j == 0:
                output_size = 19004
            else:
                output_size = 20544
        

        data = Dataset(f'../../data/subj0{i}')
        test_data = Dataset(f'../../data/subj0{i}', test=True)

        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


        # loading pretrained model
        device = torch.device("cuda")

        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        layer_names = []

        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                layer_names += [name]

        print(layer_names)

        feature_extractor = create_feature_extractor(model, 
                return_nodes=["layer3.0.conv3"]).to(device)

        feature_extractor.eval()

        regression_head = RegressionHead(output_size).to(device)
        regression_head.to(device)


        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.Adam(regression_head.parameters(), lr=1e-4)


        losses = []
        val_losses = []

        print("\nSubject " + str(i) + "\n")

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch}")
            for k, data in tqdm(enumerate(train_loader), total=len(train_loader)-1):
                optimizer.zero_grad()

                input_data = data[0]

                if j == 0:
                    label = data[1]
                else:
                    label = data[2]

                with torch.no_grad():
                    output = feature_extractor(input_data.to(device))
                
                output = regression_head(output["layer3.0.conv3"])

                loss = criterion(output, torch.tensor(label, dtype=torch.float32).to(device))
                
                if k == len(train_loader)-1:
                    losses.append(loss.item())
                    print("Train loss: " + str(loss.item()))

                loss.backward()

                del input_data
                del label
                
                optimizer.step()
        

        plt.plot(losses, label="Train loss")
        if right == True:
            plt.savefig("loss_graphs/subj0" + str(i) + "_right.jpg")
        else:
            plt.savefig("loss_graphs/subj0" + str(i) + "_left.jpg")
        

        fmri_test_pred = np.array([])

        for data in tqdm(test_loader):
            input_data = data

            with torch.no_grad():
                output = feature_extractor(input_data.to(device))
                output = regression_head(output["layer3.0.conv3"])

            if fmri_test_pred.shape[0] == 0:
                fmri_test_pred = output.detach().cpu().numpy()
            else:
                fmri_test_pred = np.vstack((fmri_test_pred, output.detach().cpu().numpy()))
            
            del input_data


        fmri_test_pred = fmri_test_pred.astype(np.float32)
        np.save(os.path.join(f"algonauts_2023_challenge_submission/subj0{i}/", 'rh_pred_test.npy' if right else 'lh_pred_test.npy'), fmri_test_pred)

        torch.cuda.empty_cache()
        gc.collect()