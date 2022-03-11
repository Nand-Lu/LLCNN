import torch
import torch.nn as nn
from math import sqrt


class LLCNN(nn.Module):
    def __init__(self):
        super(LLCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1,groups=1)

        self.conv3x1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,groups=1)

        self.conv1x1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0,groups=1)

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, groups=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_relu = self.relu(conv1)
        conv2 = self.conv1x1(conv1_relu)
        conv3 = self.conv3x1(conv1_relu)
        conv3_relu = self.relu(conv3)
        conv4 = self.conv3x1(conv3_relu)
        res2a = torch.add(conv2, conv4)
        res2a_relu = self.relu(res2a)
        conv5 = self.conv3x1(res2a_relu)
        conv5_relu = self.relu(conv5)
        conv6 = self.conv3x1(conv5_relu)
        res2b = torch.add(res2a_relu, conv6)
        res2b_relu = self.relu(res2b)
        conv7 = self.conv1x1(res2b_relu)
        conv8 = self.conv3x1(res2b_relu)
        conv8_relu = self.relu(conv8)
        conv9 = self.conv3x1(conv8_relu)
        res3a = torch.add(conv7, conv9)
        res3a_relu = self.relu(res3a)
        conv10 = self.conv3x1(res3a_relu)
        conv10_relu = self.relu(conv10)
        conv11 = self.conv3x1(conv10_relu)
        res3b = torch.add(res3a_relu, conv11)
        res3b_relu = self.relu(res3b)
        conv12 = self.conv1x1(res3b_relu)
        conv13 = self.conv3x1(res3b_relu)
        conv13_relu = self.relu(conv13)
        conv14 = self.conv3x1(conv13_relu)
        res4a = torch.add(conv12, conv14)
        res4a_relu = self.relu(res4a)
        conv15 = self.conv3x1(res4a_relu)
        conv15_relu = self.relu(conv15)
        conv16 = self.conv3x1(conv15_relu)
        res4b = torch.add(res4a_relu, conv16)
        res4b_relu = self.relu(res4b)
        conv17 = self.conv2(res4b_relu)



        return conv17
