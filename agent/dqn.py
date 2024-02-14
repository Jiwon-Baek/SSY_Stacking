import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from agent.network import Network


class Agent:
    def __init__(self):
        self.network
        self.optimizer
        pass

    def get_action(self, state):
        pass

    def train(self):
        pass

    def save_model(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode%d.pt" % e)