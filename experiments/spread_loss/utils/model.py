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
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        self.bnorm1 = nn.BatchNorm2d(256)
        self.bnorm2 = nn.BatchNorm2d(256)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.bnorm4 = nn.BatchNorm2d(256)

        # final layers
        self.final_conv1 = nn.Conv2d(1024, 512, (3, 3), bias=False)
        self.bnorm5 = nn.BatchNorm2d(512)
        self.final_conv2 = nn.Conv2d(512, 512, (3, 3), bias=False)
        self.bnorm6 = nn.BatchNorm2d(512)
        self.final_conv3 = nn.Conv2d(512, 512, (3, 3), bias=False)
        self.bnorm7 = nn.BatchNorm2d(512)

        # flatten layers creates large number of features
        # self.flatten = nn.Flatten()

        self.aggregate = nn.AvgPool2d((18, 18)) # global feature aggregation for regression output connection

        self.dense1 = nn.Linear(512, 512)
        self.dense2 = nn.Linear(512, num_vertices)

        self.gelu = GELU()

    def forward(self, x):
        out1 = self.poolx2(self.gelu(self.bnorm1(x["layer1.1.conv1"])))
        out2 = self.poolx2(self.gelu(self.bnorm2(x["layer2.0.conv1"])))
        out3 = self.gelu(self.bnorm3(x["layer3.0.conv1"]))
        out4 = self.upsamplex2(self.gelu(self.bnorm4(x["layer4.0.conv1"])))

        concat_output = torch.cat((out1, out2, out3, out4), 1)

        final = self.gelu(self.bnorm5(self.final_conv1(concat_output)))
        final = self.gelu(self.bnorm6(self.final_conv2(final)))
        final = self.gelu(self.bnorm7(self.final_conv3(final)))

        final = self.aggregate(final).squeeze(2).squeeze(2)
        final = self.gelu(self.dense1(final))

        # final linear output
        final = self.dense2(final)

        return final