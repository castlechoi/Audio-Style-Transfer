import torch.nn as nn
class VGG19(nn.Module):
    """ Can modified VGG model of channel parameters by using config data """
    def __init__(self,config):
        super(VGG19,self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                config.in_channel, 
                config.conv_1_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_1_channel,
                config.conv_1_channel, 
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2),stride = 1)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                config.conv_1_channel,
                config.conv_2_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_2_channel,
                config.conv_2_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = 1 )
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                config.conv_2_channel,
                config.conv_3_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_3_channel,
                config.conv_3_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_3_channel,
                config.conv_3_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_3_channel,
                config.conv_3_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = 1)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(
                config.conv_3_channel,
                config.conv_4_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_4_channel,
                config.conv_4_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_4_channel,
                config.conv_4_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_4_channel,
                config.conv_4_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = 1)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(
                config.conv_4_channel,
                config.conv_5_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_5_channel,
                config.conv_5_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_5_channel,
                config.conv_5_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_5_channel,
                config.conv_5_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = 1)
        )
        
    def forward(self, x):
        x = x.view(3,1,128,128)
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        return out
        
class VGGContent(nn.Module):
    def __init__(self,config):
        super(VGGContent, self).__init__()
        self.conv_4 = nn.Sequential(
            nn.Conv2d(
                1,
                config.conv_4_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_4_channel,
                config.conv_4_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_4_channel,
                config.conv_4_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                config.conv_4_channel,
                config.conv_4_channel,
                kernel_size = 3, stride = 1, padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = 1)
        )
    def forard(self,x):
        return self.conv_4(x)