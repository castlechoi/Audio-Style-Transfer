import torch
import torch.nn as nn
from VGGModels import VGG19
class NeuralTransfer(nn.Module):
    def __init__(self, config):
        super(NeuralTransfer,self).__init__()
        self.vgg = VGG19(config)
        
    def forward(self, x):
        return self.vgg(x) # batch_size * 256 * W * H