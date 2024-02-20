import os
import json
import torch

from datetime import datetime

from cfg_train import get_cfg
# from agent.DeepSARSA import *
from agent import DQN, DeepSARSA, DQN_2015
from environment.data import DataGenerator
from environment.env import *
import matplotlib.pyplot as plt
from environment.GUI import GUI
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)


def evaluate(val_dir):
    val_paths = os.listdir(val_dir)


def train(n_episode, type_network, _num_piles, _num_plates, _max_height,
          _epsilon_decay, _learning_rate, _optimizer_type, _reward_heuristic):
    torch.manual_seed(42)
    # print(torch.randn(2))
    # print(torch.randn(2))
    # print(torch.randn(2))
    now = datetime.now()
    filename = now.strftime('%Y-%m-%d-%H-%M-%S')
    model_dir = './result/train/' + type_network + '/' + filename + '/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = './result/train/' + type_network + '/' + filename + '/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    num_piles = _num_piles
    num_plates = _num_plates
    max_height = _max_height

    data_generator = DataGenerator(num_plates)
    env = Stacking(data_generator, num_piles=num_piles, max_height=max_height,
                   device=device, reward_heuristic=_reward_heuristic)
    state_size = (max_height, (num_piles + 1))
    if type_network == 'DQN':
        agent = DQN.Agent(state_size, num_piles, device)
    else:
        agent = DeepSARSA.Agent(state_size, num_piles, device)

    agent.set_params(_epsilon_decay, _learning_rate, _optimizer_type)
    scores, episodes, epsilons, crane_moves = [], [], [], []

    EPISODES = n_episode
    for e in range(EPISODES):
        # Initialize
        done = False
        score = 0
        state = env.reset()  # (max_height, num_piles+1)
        max_score = 0.0
        while not done:
            action = agent.get_action(state)  # (int)
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state)
            agent.train(state, action, reward, next_state, next_action, done)
            score += reward
            state = next_state
            if done:
                if e % 50 == 49:
                    print("episode:{0} \tscore:{1} \tepsilon: {2} \tcrane_move:{3}".format(
                    e+1, round(score, 3), round(agent.epsilon, 3), env.crane_move))
                scores.append(score)
                crane_moves.append(env.crane_move*2)
                if score > max_score:
                    max_score = score
                episodes.append(e)
                epsilons.append(agent.epsilon * 100.0)
    plt.figure()
    plt.plot(episodes, scores, label='score')
    plt.plot(episodes, epsilons, label='epsilon(1:20)')
    plt.plot(episodes, crane_moves, label='crane move(1:2)')
    title = type_network + '_decay_' + str(_epsilon_decay) \
            + '_lr_' + str(_learning_rate) + '_'+_optimizer_type+'_'\
            + '_heuristic' + str(_reward_heuristic)
    plt.title(type_network + ', decay:' + str(_epsilon_decay) \
            + ', lr:' + str(_learning_rate) + ', '+_optimizer_type \
            + ', heuristic:' + str(_reward_heuristic))
    plt.legend()
    plt.savefig('./result/train/' + type_network + '/' + filename + '/'+title+'.png')
    plt.savefig('./result/train/'+title+'.png')



if __name__ == "__main__":
    date = datetime.now().strftime('%m%d_%H_%M')

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

    # with open(log_dir + "parameters.json", 'w') as f:
    #     json.dump(vars(cfg), f, indent=4)

    data_generator = DataGenerator(num_plates)
    env = Stacking(data_generator, num_piles=num_piles, max_height=max_height,
                   device=device, reward_heuristic=1)
    state_size = (max_height, (num_piles + 1))

    state = np.random.normal(0, 1, state_size)

    agent = DQN_2015.Agent(state_size, num_piles, device)
    # agent = DQN.Agent(state_size, num_piles, device)

    scores, episodes, epsilons, crane_moves = [], [], [], []

    EPISODES = 100
    for e in range(EPISODES):
        # Initialize
        done = False
        score = 0
        state = env.reset()  # (max_height, num_piles+1)
        max_score = 0.0
        while not done:
            action = agent.get_action(state)  # (int)
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state)
            agent.train(state, action, reward, next_state, next_action, done)
            score += reward
            state = next_state
            if done:
                print("episode:{0} \tscore:{1} \tepsilon: {2} \tcrane_move:{3}".format(
                    e, round(score, 3), round(agent.epsilon, 3), env.crane_move))
                scores.append(score)
                crane_moves.append(env.crane_move)
                if score > max_score:
                    max_score = score
                    # agent.change_optimizer()

                episodes.append(e)
                epsilons.append(agent.epsilon * 100.0)

    plt.figure()
    plt.plot(episodes, scores, label='score')
    plt.plot(episodes, epsilons, label='epsilon(1:100)')
    plt.plot(episodes, crane_moves, label='crane move')
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
