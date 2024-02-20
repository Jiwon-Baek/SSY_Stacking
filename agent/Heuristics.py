import torch
import numpy as np
from environment.data import DataGenerator
from environment.env import Stacking
from environment.GUI import *
class Agent:
    def __init__(self, env):
        # MDP
        self.env = env

    def get_action(self):
        date = self.env.plates[0].retrieval_date
        selected = None
        selected_gap = 1e8 # trivial number that is big enough
        empty = []
        for i, pile in enumerate(self.env.piles):
            if len(pile): # pile에 한 개 이상 plate가 존재하면
                recent = pile[-1]
                # 새 plate의 꺼내는 날짜가 더 이전이면
                gap = recent.retrieval_date - date
                if gap >=0:
                    if gap <= selected_gap:
                        # 만약 출고 날짜의 간격이 더 짧은 pile의 plate를 찾는다면, 수정
                        selected = i
                        selected_gap = gap
            else:
                empty.append(i) # 빈 pile들의 목록
        if selected is None: # 고를 수 있었던 pile이 하나도 없으면
            if empty: # 그 와중에 빈 파일이 존재하면
                return np.random.choice(empty)
            else:
                return np.random.choice(len(self.env.piles))
        return selected

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_episode = 10
    load_model = 0

    num_piles = 10
    num_plates = 20
    max_height = 20

    log_every = 10
    eval_every = 100
    save_every = 1000
    new_instance_every = 10

    data_generator = DataGenerator(num_plates)
    env = Stacking(data_generator, num_piles=num_piles, max_height=max_height,
                   device=device, reward_heuristic=1)
    agent = Agent(env)

    # show the final result
    done = False
    score = 0
    state = env.reset()  # (max_height, num_piles+1)
    images = []
    while not done:
        action = agent.get_action()  # (int)
        next_state, reward, done = env.step(action)
        score += reward
        images.append(env.show_state(next_state))
        if done:
            print('Heuristic Stacking Result')
            print('Score : ',score)
            print('Crane Move : ',env.crane_move)

            gui = GUI(images)
