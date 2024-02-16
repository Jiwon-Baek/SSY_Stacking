import os
import json
import torch

from datetime import datetime

from cfg_train import get_cfg
# from agent.DeepSARSA import *
from agent.DQN import *
from environment.data import DataGenerator
from environment.env import *
import matplotlib.pyplot as plt
from environment.GUI import GUI
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
def evaluate(val_dir):
    val_paths = os.listdir(val_dir)



if __name__ == "__main__":
    date = datetime.now().strftime('%m%d_%H_%M')
    cfg = get_cfg()

    # n_episode = cfg.n_episode
    # load_model = cfg.load_model
    #
    # num_piles = cfg.num_piles
    # num_plates = cfg.num_plates
    # max_height = cfg.max_height
    #
    # log_every = cfg.log_every
    # eval_every = cfg.eval_every
    # save_every = cfg.save_every
    # new_instance_every = cfg.new_instance_every

    n_episode = 10
    load_model = 0

    num_piles = 10
    num_plates = 20
    max_height = 20

    log_every = 10
    eval_every = 100
    save_every = 1000
    new_instance_every = 10

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
    state_size = (max_height , (num_piles + 1))

    state = np.random.normal(0, 1, state_size)

    agent = Agent(state_size, num_piles)

    scores, episodes, epsilons = [], [], []

    EPISODES = 200
    for e in range(EPISODES):
        # Initialize
        done = False
        score = 0
        state = env.reset() # (max_height, num_piles+1)
        max_score = 0.0
        while not done:
            action = agent.get_action(state)  # (int)
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state)
            agent.train(state,action,reward, next_state, next_action, done)
            score += reward
            state = next_state
            if done:
                print("episode:{0} \tscore:{1} \tepsilon: {2} \tcrane_move:{3}".format(
                    e,round(score,3),round(agent.epsilon,3),env.crane_move))
                scores.append(score)
                if score > max_score:
                    max_score = score
                    # agent.change_optimizer()

                episodes.append(e)
                epsilons.append(agent.epsilon*100.0)

    plt.figure()
    plt.plot(episodes, scores, label='score')
    plt.plot(episodes, epsilons, label='epsilon(1:20)')
    plt.title('DQN')
    plt.legend()
    plt.show()

    # show the final result
    done = False
    score = 0
    state = env.reset()  # (max_height, num_piles+1)
    images = []
    while not done:
        action = agent.get_action(state)  # (int)
        next_state, reward, done = env.step(action)
        next_action = agent.get_action(next_state)
        agent.train(state, action, reward, next_state, next_action, done)
        score += reward
        state = next_state
        images.append(env.show_state(state))
        if done:
            gui = GUI(images)





