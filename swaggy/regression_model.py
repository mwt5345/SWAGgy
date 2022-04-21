import torch
import torchvision

class Resnet18Regression(torch.nn.Module):
    def __init__(self, num_of_input_channels, output_size):
        super(Resnet18Regression, self).__init__()
        # load in resnet18 model
        self.resnet18 = torchvision.models.resnet18()
        # modify the number of channels
        self.resnet18.conv1 = torch.nn.Conv2d(num_of_input_channels,64,kernel_size=(7, 7),
            stride=(2, 2), padding=(3, 3), bias=False)
        # modify the number of outputs
        self.resnet18.fc = torch.nn.Linear(in_features=512,out_features=output_size,bias=True)

    def forward(self, x):
        out = self.resnet18(x)
        return out