from train import train
import torch
import pandas as pd
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

# tested_set = pd.DataFrame(columns = ['RL', 'E_decay', 'Learning Rate', 'Optimizer', 'Reward Heuristic'])
tested_set = pd.read_csv('./result/tested_parameters.csv', index_col=0)


for t in ['DQN', 'DeepSARSA']:
# for t in ['DQN']:
    # for e_decay in [0.999, 0.99]:
    for e_decay in [0.9995]:
        # for lr in [1e-3, 1e-4]:
        # for lr in [1e-5, 1e-6]:
        for lr in [1e-4, 1e-5]:

            
            # for opt in ['SGD', 'Adam']:
            for opt in ['SGD']:
                for heuristic in range(1,7):
                    tested_set.loc[tested_set.shape[0]] = [t, e_decay, lr, opt, heuristic]
                    train(2000, t, 10, 20, 20, e_decay, lr, opt, heuristic)
tested_set.to_csv('./result/tested_parameters.csv')
# for t in range(2):
    # torch.manual_seed(42)

