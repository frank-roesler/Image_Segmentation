import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50


class AutoEncoder(nn.Module):
    """Implementation of basic convolutional autoencoder architecture.
    nc specifies the number of channels that increases after every encoder layer and decreases again in the decoder."""
    def __init__(self, nc=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, nc, kernel_size=(4,4), stride=(2,2), padding=(1,1)), # RGB = 3 input channels
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, 2*nc, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(2*nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*nc, 4*nc, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(4*nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*nc, 8*nc, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(8*nc),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8*nc, 4*nc, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(4*nc),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4*nc, 2*nc, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(2*nc),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*nc, nc, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nc, nc, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, 1, kernel_size=(1,1), stride=(1,1), padding=(0,0)),  # 1 output channel (for 2 classes)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FCN_Resnet(nn.Module):
    """fully convolutional version of ResNet architecture. __init__ function loads the model and replaces the last layer
    in order to obtain 1 output channel (for 2-class segmentation)."""
    def __init__(self):
        super().__init__()
        self.resnet = fcn_resnet50(pretrained=True)
        self.resnet.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x['out'])











