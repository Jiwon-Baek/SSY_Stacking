import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random


class SarsaNet(nn.Module):
    def __init__(self, state_size, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.state_size = state_size
        self.conv1 = nn.Conv2d(1,32,kernel_size=3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.input_dim = 64 * state_size[0] * state_size[1]
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, self.output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, state):
        x = torch.tensor(state.copy(), dtype=torch.float32).view(1,1,self.state_size[0],self.state_size[1])
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x).view(1,-1)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        # x = F.relu(self.fc2(x))
        x = self.fc2(x) # 출력층에는 relu가 없는게 나을까?
        # print(x.size())
        return x

class Agent:
    def __init__(self, state_size, action_size):

        # MDP
        self.state_size = state_size
        self.action_size = action_size

        # SARSA parameters
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

        # learning parameters
        self.model = SarsaNet(state_size, self.action_size)
        self.learning_rate = 1e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    def change_optimizer(self):
        self.learning_rate = self.learning_rate * 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

    def get_action(self, state):
        # print(random.randrange(self.action_size))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            pred_q = self.model(state)

            # print("pred_q", pred_q)
            # print("torch.argmax(pred_q).item()", torch.argmax(pred_q).item())
            return torch.argmax(pred_q).item()

    def train(self, s, a, r, next_s, next_a, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.model.train()
        self.optimizer.zero_grad()
        q = self.model(s)[0]
        max_q = torch.amax(q, axis=-1)

        target = r + (1-done) * self.discount_factor * max_q
        loss_function = nn.MSELoss()

        loss = loss_function(target, q)
        loss.backward()
        self.optimizer.step()


    # def save_model(self, e, file_dir):
    #     torch.save({"episode": e,
    #                 "model_state_dict": self.network.state_dict(),
    #                 "optimizer_state_dict": self.optimizer.state_dict()},
    #                file_dir + "episode%d.pt" % e)

if __name__ == "__main__":
    max_height = 10
    num_piles = 10
    state_size = (max_height, (num_piles+1))
    state = np.random.normal(0, 1, state_size)
    agent = Agent(state_size, num_piles)
    agent.get_action(state)

