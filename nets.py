import torch.nn as nn
import torch
import torch.nn.functional as F


class MyFCN(nn.Module):
    """Implementation of FCN-Alexnet, introduced in https://arxiv.org/abs/1411.4038"""
    def __init__(self):
        super(MyFCN, self).__init__()
        self.down_features = nn.Sequential(
            nn.Conv2d(3,96, kernel_size=11, stride=4, padding=100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.up_features = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(256, 4096, kernel_size=6, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 1, kernel_size=1, padding=0),
            nn.ConvTranspose2d(1, 1, kernel_size=63, stride=32, bias=False)
        )
    #     The 1, 1 in the last layer is the number of classes (except background) and can be changed to any other
    #     number with slight modifications to the rest of the code.

    def center_crop(self, x, height, width):
        crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
        crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)

        return F.pad(x, [
            crop_w.ceil().int()[0], crop_w.floor().int()[0],
            crop_h.ceil().int()[0], crop_h.floor().int()[0],
        ])

    def forward(self, x):
        down = self.down_features(x)
        up  = self.up_features(down)
        out = self.center_crop(up, *x.shape[2:4])
        out = torch.sigmoid(out)
        return out
















