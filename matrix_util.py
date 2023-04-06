import torch.nn as nn
class GramMatrix(nn.Module):
	def forward(self, input):
		a, b, c = input.size()
		features = input.view(a * b, c) 
		G = torch.mm(features, features.t())
		return G.div(a * b * c)