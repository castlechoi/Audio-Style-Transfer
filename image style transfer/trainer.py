import torch
import torch.nn
from loss import StyleLoss, ContentLoss

from model import GramMaxtirx
def Trainer(self, model, optimizer, style_img, content_img, config):
    criterion_style = StyleLoss()
    criterion_content = ContentLoss()
    
    for i in range(config.epochs + 1):
        optimizer.zero_grad()
        content_pred = model(content_img)
        style_pred = model(style_img)
        
        loss_style = criterion(content_img)
        loss_content = criterion(style_img)
    
        loss = loss_style + loss_content
        loss.backward()
        
        optimizer.step()
        
        print(f'Loss : {loss.item():.6f}')
        
    with torch.no_grad():
        content_pred = model(content_img)
        style_pred = model(style_img)
        
        mix_pred = (content_pred + style_pred) / 2
        
        return content_pred, style_pred, mix_pred
    