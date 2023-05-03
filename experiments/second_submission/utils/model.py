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
        self.poolx4 = nn.MaxPool2d(4)
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        self.bnorm1 = nn.BatchNorm2d(512)
        self.bnorm2 = nn.BatchNorm2d(512)
        self.bnorm3 = nn.BatchNorm2d(512)
        self.bnorm4 = nn.BatchNorm2d(512)

        # final layers
        self.final_conv1 = nn.Conv2d(2048, 2048, (3, 3), bias=False)
        self.bnorm5 = nn.BatchNorm2d(2048)
        self.final_conv2 = nn.Conv2d(2048, 2048, (3, 3), bias=False)
        self.bnorm6 = nn.BatchNorm2d(2048)
        self.final_conv3 = nn.Conv2d(2048, 2048, (3, 3), bias=False)
        self.bnorm7 = nn.BatchNorm2d(2048)
        self.final_conv4 = nn.Conv2d(2048, 2048, (3, 3), bias=False)
        self.bnorm8 = nn.BatchNorm2d(2048)

        self.aggregate = nn.AvgPool2d((6, 6)) # global feature aggregation for regression output connection

        self.dense1 = nn.Linear(2048, 4096)
        self.dense2 = nn.Linear(4096, num_vertices)

        self.gelu = GELU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.5)

    def forward(self, x):
        out1 = self.poolx4(self.lrelu(self.bnorm1(x["layer1.2.conv3"])))
        out2 = self.poolx2(self.lrelu(self.bnorm2(x["layer2.3.conv3"])))
        out3 = self.lrelu(self.bnorm3(x["layer3.5.conv3"]))
        out4 = self.upsamplex2(self.lrelu(self.bnorm4(x["layer4.2.conv3"])))

        concat_output = torch.cat((out1, out2, out3, out4), 1)

        final = self.lrelu(self.bnorm5(self.final_conv1(concat_output)))
        final = self.lrelu(self.bnorm6(self.final_conv2(final)))
        final = self.lrelu(self.bnorm7(self.final_conv3(final)))
        final = self.bnorm8(self.final_conv4(final))

        final = self.aggregate(final).squeeze(2).squeeze(2)
        final = self.dense1(final)

        # final linear output
        final = self.dense2(final)

        return final