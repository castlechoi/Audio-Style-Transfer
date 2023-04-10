import torch
import torch.nn
from loss import StyleLoss, ContentLoss

def Trainer(model, optimizer, style_img, content_img, config):
    """ Define Loss """
    criterion_style = StyleLoss(style_img)
    criterion_content = ContentLoss(content_img)

    """ Don't know why it doesn't work """
    model.requires_grad_(False)
    criterion_style.requires_grad_(True)
    criterion_content.requires_grad_(True)

    """ Training """
    model.train()
    for i in range(config.epochs + 1):
        optimizer.zero_grad()
        content_pred = model(content_img)
        style_pred = model(style_img)
        
        loss_style = criterion_style(content_img)
        loss_content = criterion_content(style_img)
    
        loss = loss_style + loss_content
        loss.backward()
        
        optimizer.step()
        
        print(f'Epoch {i}  Loss  {loss.item():.6f}')
        
    """ Evaluation """
    model.eval()
    with torch.no_grad():
        content_pred = model(content_img)
        style_pred = model(style_img)
        
        mix_pred = (content_pred + style_pred) / 2
        
        return content_pred, style_pred, mix_pred
    