import torch.nn.functional as F
import torch.nn as nn
dropout_value = 0.1

#Net implementing Batch Norm
class BatchNormNet(nn.Module):
    def __init__(self):
        super(BatchNormNet, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_value)
        ) # output_size = 32 #rf = 30

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        ) # output_size = 32 #rf = 28

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 32
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16 ##rf = 14

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            #nn.Dropout(dropout_value)
        ) # output_size = 16 #rf = 12

        # CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            #nn.Dropout(dropout_value)
        ) # output_size = 16 #rf = 10

        # CONVOLUTION BLOCK 4
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            #nn.Dropout(dropout_value)
        ) # output_size = 16 #rf = 8

        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 16

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8 #rf = 4

        # CONVOLUTION BLOCK 5
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            #nn.Dropout(dropout_value)
        ) # output_size = 16 #rf = 12

        # CONVOLUTION BLOCK 6
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            #nn.Dropout(dropout_value)
        ) # output_size = 16 #rf = 10

        # CONVOLUTION BLOCK 7
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            #nn.Dropout(dropout_value)
        ) # output_size = 16 #rf = 8


        # GAP BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1

        # TRANSITION BLOCK 3
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.output(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
