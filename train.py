import os
import json
import torch

from datetime import datetime

from cfg_train import get_cfg
from agent.dqn import *
from environment.data import DataGenerator
from environment.env import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate(val_dir):
    val_paths = os.listdir(val_dir)



if __name__ == "__main__":
    date = datetime.now().strftime('%m%d_%H_%M')
    cfg = get_cfg()

    n_episode = cfg.n_episode
    load_model = cfg.load_model

    num_piles = cfg.num_piles
    num_plates = cfg.num_plates
    max_height = cfg.max_height

    log_every = cfg.log_every
    eval_every = cfg.eval_every
    save_every = cfg.save_every
    new_instance_every = cfg.new_instance_every

    model_dir = './output/train/' + date + '/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = './output/train/' + date + '/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_dir + "parameters.json", 'w') as f:
        json.dump(vars(cfg), f, indent=4)

    data_generator = DataGenerator(num_plates)
    env = Stacking(data_generator, num_piles=num_piles, max_height=max_height, device=device)