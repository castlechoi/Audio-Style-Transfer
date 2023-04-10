import torch.nn as nn
import torch
"""
        a : batch size
        b : feature map count
        (c,d) : feature map dim
        
        gram_matrix = A dot A_transpose
"""
def gram_matrix(input):
    a,b,c,d = input.size()
    features = input.view(a*b, c*d) # flatten feature map
    
    G = torch.mm(features, features.t()) # output -> a*b a*b
    
    # normalized ->  값의 개수 ? 
    return G.div(a*b*c*d)
    
class ContentLoss(nn.Module):
    """ Loss is calculated based on original image which is saved in ContentLoss object """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.criterion = nn.MSELoss()
        
    def forward(self, input):
        loss = self.criterion(input, self.target)
        return loss
    
class StyleLoss(nn.Module):
    """ gram matrix loss between two images """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.criterion = nn.MSELoss()
        
    def forward(self, input):
        G = gram_matrix(input)
        loss =  self.criterion(G, self.target)
        return loss
    
      
     