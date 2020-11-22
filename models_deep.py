## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:

        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel

        # Input 224X224
        self.conv1 = nn.Conv2d(1, 32, 5) # Convolutional layer 1
        self.conv1_bn = nn.BatchNorm2d(32)
        # Output size = (Width-FKernel)/Stride +1 = (32, 220, 220)
        # Layes output = (32, 220, 220)

        self.conv2 = nn.Conv2d(32, 32, 5) # Convolutional layer 2
        self.conv2_bn = nn.BatchNorm2d(32)
        # Output size = (Width-FKernel)/Stride +1 = (32, 216, 216)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Layes output = (32, 108, 108)

        self.conv3 = nn.Conv2d(32, 64, 5) # Convolutional layer 3
        self.conv3_bn = nn.BatchNorm2d(64)
        # Output size = (Width-FKernel)/Stride +1 = (64, 104, 104)

        self.conv4 = nn.Conv2d(64, 64, 5) # Convolutional layer 4
        self.conv4_bn = nn.BatchNorm2d(64)
        # Output size = (Width-FKernel)/Stride +1 = (64, 100, 100)
        self.pool4 = nn.MaxPool2d(2, 2) 
        # Layes output = (64, 50, 50)

        self.conv5 = nn.Conv2d(64, 128, 5) # Convolutional layer 5
        self.conv5_bn = nn.BatchNorm2d(128)
        # Output size = (Width-FKernel)/Stride +1 = (128, 46, 46)

        self.conv6 = nn.Conv2d(128, 128, 5) # Convolutional layer 6
        self.conv6_bn = nn.BatchNorm2d(128)
        # Output size = (Width-FKernel)/Stride +1 = (128, 42, 42)
        self.pool6 = nn.MaxPool2d(2, 2)
        # Layes output = (128, 21, 21)

        self.conv7 = nn.Conv2d(128, 256, 2) # Convolutional layer 7
        self.conv7_bn = nn.BatchNorm2d(256)
        # Output size = (Width-FKernel)/Stride +1 = (256, 20, 20)

        self.conv8 = nn.Conv2d(256, 256, 3) # Convolutional layer 8
        self.conv8_bn = nn.BatchNorm2d(256)
        # Output size = (Width-FKernel)/Stride +1 = (256, 18, 18)
        self.pool8 = nn.MaxPool2d(2, 2)
        # Layes output = (256, 9, 9)

        self.conv9 = nn.Conv2d(256, 512, 2) # Convolutional layer 9
        self.conv9_bn = nn.BatchNorm2d(512)
        # Output size = (Width-FKernel)/Stride +1 = (512, 8, 8)

        self.conv10 = nn.Conv2d(512, 512, 3) # Convolutional layer 10
        self.conv10_bn = nn.BatchNorm2d(512)
        # Output size = (Width-FKernel)/Stride +1 = (128, 6, 6)
        self.pool10 = nn.MaxPool2d(2, 2)
        # Layes output = (512, 3, 3)

        self.fc11 = nn.Linear(512*3*3, 1024) # Dense layer 1
        self.drop11 = nn.Dropout(p=0.15, inplace=True)

        self.fc12 = nn.Linear(1024, 136) # Dense layer output


        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers,
        # and other layers (such as dropout or batch normalization) to avoid overfitting



    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), inplace=False)
        x = self.pool2(F.leaky_relu(self.conv2_bn(self.conv2(x)), inplace=False))
        
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), inplace=False)
        x = self.pool4(F.leaky_relu(self.conv4_bn(self.conv4(x)), inplace=False))
        
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), inplace=False)
        x = self.pool6(F.leaky_relu(self.conv6_bn(self.conv6(x)), inplace=False))

        x = F.leaky_relu(self.conv7_bn(self.conv7(x)), inplace=False)
        x = self.pool8(F.leaky_relu(self.conv8_bn(self.conv8(x)), inplace=False))

        x = F.leaky_relu(self.conv9_bn(self.conv9(x)), inplace=False)
        x = self.pool10(F.leaky_relu(self.conv10_bn(self.conv10(x)), inplace=False))

        x = x.view(x.size(0), -1) #reshape for the FC layer

        x = self.drop11(F.leaky_relu(self.fc11(x), inplace=False))
        x = self.fc12(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
