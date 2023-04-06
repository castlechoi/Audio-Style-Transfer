import os
import argparse
import numpy as np

from datetime import datetime

from model import CNN
from dataloader import data_generator
from trainer import Trainer


# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

# Input Args
home_dir = os.getcwd()
parser.add_argument('--style', default='style.wav', type=str, help='Style data')
parser.add_argument('--content', default='content.wav', type=str, help='Content data')
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

# load model
style_transfer_model = CNN()

data_path = "./dataset"

style_wv, style_sr,content_wv, content_wv = data_generator(os.path.join(data_path, "style.wav"),os.path.join(data_path, "content.wav"))


Trainer()
