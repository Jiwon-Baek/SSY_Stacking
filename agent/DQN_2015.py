"""
https://ai-com.tistory.com/entry/RL-강화학습-알고리즘-1-DQN-Deep-Q-Network
"""
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from collections import namedtuple, deque

BATCH_SIZE = 4

class DQNnet(nn.Module):
    def __init__(self, state_size, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.state_size = state_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
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
        x = state.view(1, 1, self.state_size[0], self.state_size[1]).type(torch.float32)
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x).view(1, -1)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)  # 출력층에는 relu가 없는게 나을까?
        # print(x.size())
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)
        # collections 라이브러리의 deque를 사용하면 일정한 크기를 갖는 메모리 생성 가능
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class Agent:
    def __init__(self, state_size, action_size, device):

        # MDP
        self.state_size = state_size
        self.action_size = action_size

        # RL parameters
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.replay_memory = []
        self.update_period = 4
        self.num_step = 0
        self.num_sample = 4
        self.memory = ReplayMemory(capacity=10000)

        # learning parameters
        self.main_net = DQNnet(state_size, self.action_size).to(device)
        self.target_net = copy.deepcopy(self.main_net)
        self.learning_rate = 1e-4
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.learning_rate)

    def set_params(self, _epsilon_decay, _learning_rate, _optimizer_type):
        self.epsilon_decay = _epsilon_decay
        # learning parameters
        self.learning_rate = _learning_rate
        if _optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.main_net.parameters(), lr=self.learning_rate)
        elif _optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.main_net.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.learning_rate)

    def reset_memory(self):
        self.replay_memory = []



    def get_action(self, state):
        # print(random.randrange(self.action_size))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            pred_q = self.main_net(state)

            # print("pred_q", pred_q)
            # print("torch.argmax(pred_q).item()", torch.argmax(pred_q).item())
            return torch.argmax(pred_q).item()

    def train(self, s, a, r, next_s, next_a, done):



        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.main_net.train()
        self.optimizer.zero_grad()
        # 마지막 출력층 개수인 10과 실제 MSE 계산값의 차원(())이 안 맞아서 문제가 생기는 듯?
        self.replay_memory.append({'state':s, 'action':a, 'reward':r, 'next state':next_s})

        # if len(self.memory)<BATCH_SIZE:
        #     return
        # transitions = self.memory.sample(BATCH_SIZE)
        # batch = Transition(*zip(*transitions))

        # Experience Replay
        indices = torch.randint(0, len(self.replay_memory), (self.num_sample,))
        target_list = torch.zeros(self.num_sample,)
        action_value_list = torch.zeros(self.num_sample,)

        for i, idx in enumerate(indices):
            state = self.replay_memory[idx]['state']
            action = self.replay_memory[idx]['action']
            next_state = self.replay_memory[idx]['next state']
            reward = self.replay_memory[idx]['reward']

            action_value = self.main_net(state)[0][action].type(torch.float32)
            target_value = self.target_net(next_state)[0].type(torch.float32)
            max_q = torch.amax(target_value, axis=-1).type(torch.float32)
            target = reward + (1 - done) * self.discount_factor * max_q
            action_value_list[i] = action_value
            target_list[i] = target
        # loss_function = nn.MSELoss()
        loss_function = nn.SmoothL1Loss()
        loss = loss_function(target_list, action_value_list)
        loss.backward()
        self.optimizer.step()

        if self.num_step % self.update_period == (self.update_period -1):
            self.target_net = copy.deepcopy(self.main_net)
        self.num_step +=1

    def update_target_model(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    # def save_model(self, e, file_dir):
    #     torch.save({"episode": e,
    #                 "model_state_dict": self.network.state_dict(),
    #                 "optimizer_state_dict": self.optimizer.state_dict()},
    #                file_dir + "episode%d.pt" % e)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    max_height = 10
    num_piles = 10
    state_size = (max_height, (num_piles + 1))
    state1 = torch.normal(0, 1, state_size).to(device)
    state2 = torch.normal(0, 1, state_size).to(device)
    agent = Agent(state_size, num_piles, device)
    action1 = agent.get_action(state1)
    action2 = agent.get_action(state2)
    reward = 10
    done = False
    agent.train(state1, action1, reward, state2, action2, done)
