## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
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
        
        ### output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        

        ### output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output Tensor for one image, will have the dimensions: (128, 108, 108)
        # after one pool layer, this becomes (128, 54, 54)
        self.conv2 = nn.Conv2d(32, 128, 3)
        ### output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output Tensor for one image, will have the dimensions: (256, 52, 52)
        # after one pool layer, this becomes (256, 26, 26)
        self.conv3 = nn.Conv2d(128, 256, 3)
        ### output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output Tensor for one image, will have the dimensions: (512, 24, 24)
        # after one pool layer, this becomes (512, 12, 12)
        self.conv4 = nn.Conv2d(256, 512, 3)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        
        # 512 outputs * the 12*12 filtered/pooled map size
        self.fc1 = nn.Linear(512*12*12, 2048)
        
        # dropout with p=0.3
        self.fc1_drop = nn.Dropout(p=0.3)
        
        
        self.fc2 = nn.Linear(2048, 1024)
        # dropout with p=0.1
        self.fc2_drop = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(1024, 136)



        
        
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        # three linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
                
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
