import torch
import torch.nn
from loss import StyleLoss, ContentLoss

from model import GramMaxtirx
def Trainer(self, model, optimizer, style_wv, style_sr, content_wv, content_sr, config):
    criterion_style = StyleLoss()
    criterion_content = ContentLoss()
    
    style_2d = torch.FloatTensor(style_wv).view(1,config.width,-1)
    content_2d = torch.FloatTensor(content_wv).view(1, config.width, -1)
    
    target_style = model(style_2d)
    target_content = model(content_2d)
    
    for i in range(config.epochs + 1):
        optimizer.zero_grad()
        
        loss_style = criterion(target_style)
        loss_content = criterion(target_content)
    
        loss = loss_style + loss_content
        loss.backward()
        
        optimizer.step()
        
        print(f'Loss : {loss.item():.6f}')
    