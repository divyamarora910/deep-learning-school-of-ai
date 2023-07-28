import torch.nn.functional as F
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
                          in_channels=in_channels, out_channels=out_channels, \
                          kernel_size=3, stride=stride,\
                          padding=1, bias=False
                          )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
                          in_channels=out_channels, out_channels=out_channels, \
                          kernel_size=3, stride=1,\
                          padding=1, bias=False
                          )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x+nn.Sequential()(x)
        return x

class LayerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LayerBlock, self).__init__()
        self.inner_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,\
            kernel_size=3,stride=1,\
            padding=1,bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.res_block = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.inner_layer(x)
        r = self.res_block(x)
        out = x + r
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.prep_layer = nn.Sequential(
                              nn.Conv2d(in_channels=3,out_channels=64,\
                                        kernel_size=3,stride=1,\
                                        padding=1,bias=False,\
                              ),
                              nn.BatchNorm2d(64),\
                              nn.ReLU(),
                          )

        self.layer_1 = LayerBlock(in_channels=64, out_channels=128)
        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,out_channels=256,kernel_size=3,
                stride=1,padding=1,bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer_3 = LayerBlock(in_channels=256, out_channels=512)
        self.max_pool = nn.Sequential(nn.MaxPool2d(kernel_size=4))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
