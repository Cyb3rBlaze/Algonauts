import torch
import torch.nn as nn

import math


# pulled from Dr. Karpathy's minGPT implementation
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class RegressionHead(nn.Module):
    def __init__(self, num_vertices=2837):
        super(RegressionHead, self).__init__()
        # pooling to make sure dimensionality is the same for features across multiple layers
        self.poolx2 = nn.MaxPool2d(2)

        self.bnorm1 = nn.BatchNorm2d(512)
        #self.bnorm2 = nn.BatchNorm2d(512)

        # final layers
        self.final_conv1 = nn.Conv2d(1024, 1024, (3, 3), bias=False)
        self.bnorm5 = nn.BatchNorm2d(1024)
        # self.final_conv2 = nn.Conv2d(1024, 1024, (3, 3), bias=False)
        # self.bnorm6 = nn.BatchNorm2d(1024)
        # self.final_conv3 = nn.Conv2d(1024, 1024, (3, 3), bias=False)
        # self.bnorm7 = nn.BatchNorm2d(1024)
        # self.final_conv4 = nn.Conv2d(1024, 1024, (3, 3), bias=False)
        # self.bnorm8 = nn.BatchNorm2d(1024)
        # self.final_conv5 = nn.Conv2d(1024, 1024, (3, 3), bias=False)
        # self.bnorm9 = nn.BatchNorm2d(1024)
        # self.final_conv6 = nn.Conv2d(1024, 1024, (3, 3), bias=False)
        # self.bnorm10 = nn.BatchNorm2d(1024)

        self.aggregate = nn.AvgPool2d((26, 26)) # global feature aggregation for regression output connection

        # self.dense1 = nn.Linear(1024, 4096)
        self.dense2 = nn.Linear(1024, num_vertices)

        self.gelu = GELU()

    def forward(self, x):
        #out1 = self.gelu(self.poolx2(self.bnorm1(x["layer1.2.conv2"])))
        out1 = self.gelu(x["layer3.5.conv2"])

        # across channel dim
        # concat_output = torch.cat((out1, out2), 1)

        # final = self.gelu(self.bnorm5(self.final_conv1(concat_output)))
        # final = self.gelu(self.bnorm6(self.final_conv2(final)))
        # final = self.gelu(self.bnorm7(self.final_conv3(final)))
        # final = self.gelu(self.bnorm8(self.final_conv4(final)))
        # final = self.gelu(self.bnorm9(self.final_conv5(final)))
        # final = self.gelu(self.bnorm10(self.final_conv6(final)))

        final = self.aggregate(final).squeeze(2).squeeze(2)
        # final = self.dense1(final)

        # final linear output
        final = self.dense2(final)

        return final