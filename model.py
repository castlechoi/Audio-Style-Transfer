import torch
import torch.nn as nn

class CNN(nn.Module):
	def __init__(self):
		super(CNNModel, self).__init__()
		self.cnn1 = nn.Sequential(
				nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
			 	nn.ReLU())
		self.cnn2 = nn.Sequential(
				nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1),
			 	nn.ReLU()
		)
	def forward(self, x):
		out = self.cnn1(x)
		return out