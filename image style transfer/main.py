import os
import argparse
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import data_generator
from trainer import Trainer

from config import Config


# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

# Input Args
home_dir = os.getcwd()
parser.add_argument('--style', default='dancing.png', type=str, help='Style data')
parser.add_argument('--content', default='picasso.png', type=str, help='Content data')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--seed', default=42, type=int, help='seed value')

# set GPU option
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

style_data = args.style
content_data = args.content

# fix seed number
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

config = Config()

# image style transfer
style_path = "./dataset/dancing.png"
content_path = "./dataset/picasso.png"

style_img = read_image(style_path).to(device)
content_img = read_image(content_path).to(device)



# load model
model = NeuralTransfer(config)
optimizer = optim.Adam(model.parameters(), lr = config.lr, betas = (config.beta1, config.beta2), weight_decay = config.weight_decay)


style_pred, content_pred, mix_pred = Trainer(model, optimizer, style_img, content_img, config)
