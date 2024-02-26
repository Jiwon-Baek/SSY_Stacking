import random
import numpy as np
import pandas as pd
import torch
import io
from time import sleep  # Import the sleep function
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
from environment.data import DataGenerator
# from GUI import GUI
class Plate:
    def __init__(self, id, arrival_date, retrieval_date):
        self.id = id
        self.arrival_date = arrival_date
        self.retrieval_date = retrieval_date


class Stacking:
    def __init__(self, data_src, num_piles=4, max_height=4, device=None, reward_heuristic=None):
        self.data_src = data_src
        self.num_piles = num_piles
        self.max_height = max_height  # 한 파일에 적치 가능한 강재의 수
        self.device = device
        self.reward_heuristic = reward_heuristic

        if type(self.data_src) is DataGenerator:
            self.df = self.data_src.generate()
        else:
            self.df = pd.read_excel(data_src, engine='openpyxl')

        self.piles = [[] for _ in range(num_piles)]
        self.plates = []
        for i, row in self.df.iterrows():
            id = row["plate id"]
            arrival_date = row["arrival date"]
            retrieval_date = row["retrieval date"]
            plate = Plate(id, arrival_date, retrieval_date)
            self.plates.append(plate)

        # arrival_date : 0.0
        # retrieval_date : random integer
        self.stage = 0
        self.current_date = 0
        self.crane_move = 0

    def step(self, action):

        done = False # 에피소드가 종료되었는지를 나타내는 플래그
        # done이 True일 때, 에이전트는 보상을 받고 (reward), 새로운 에피소드를 시작하기 위해 초기 상태로 리셋(reset() 메서드 호출).
        plate = self.plates.pop(0)

        if len(self.piles[action]) == self.max_height:
            done = True
            reward = -1.0
        else:
            self.piles[action].append(plate)  # action 에 따라서 강재를 적치
            reward = self._calculate_reward(action)  # 해당 action 에 대한 보상을 계산

        if len(self.plates) == 0:
            done = True
        elif self.plates[0].arrival_date != self.current_date:
            self.current_date = self.plates[0].arrival_date
            self._retrieve_plates()
            # print('retrieved plates!')

        next_state = self._get_state()  # 쌓인 강재들 리스트에서 state 를 계산

        if done:
            self._retrieve_all_plates()
            # print('retrieved all plates!')


        return next_state, reward, done

    def reset(self):
        # print(self.plates)
        # self.plates : list of Plate object
        # self.plates[0].arrival_date : float64 value
        self.plates = []
        for i, row in self.df.iterrows():
            id = row["plate id"]
            arrival_date = row["arrival date"]
            retrieval_date = row["retrieval date"]
            plate = Plate(id, arrival_date, retrieval_date)
            self.plates.append(plate)
        self.current_date = min(self.plates, key=lambda x: x.arrival_date).arrival_date
        self.crane_move = 0

        return self._get_state()

    def _calculate_reward(self, action):
        """
        reward heuristic 1: # 고정값 = 0 | 10/max_move
        reward heuristic 2: # 고정값 = 0 | 10/total_move
        reward heuristic 3: # 고정값 = 2 | 10/max_move
        reward heuristic 4: # 고정값 = 2 | 10/total_move
        reward heuristic 5: # penalty | 10/max_move
        reward heuristic 6: # penalty | 10/total_move
        reward heuristic 7: # penalty - max_move
        reward heuristic 8: # penalty - total_move
        """
        # heuristic적인 penalty를 부여하는 게 낫나?
        # 아니면 그냥 전반적인 상황을 고려하도록 하는게 낫나?
        # 오히려 penalty를 주니까 학습이 잘 안되는 것 같다


        """ 1. 만약 새로운 pile에 적재를 시작했다면, 다른 모든 pile 중에는 대안이 없었는가?"""
        pile = self.piles[action]


        max_move = 0
        total_move = 0
        penalty = 0.0

        if len(pile) == 1:
            newplate = pile[-1]
            for i, p in enumerate(self.piles):
                if i != action:  # 이번에 올린 pile 말고 다른 pile들을 대상으로 검사
                    if len(p) >= 1:  # 한 개 이상 쌓인 pile들을 대상으로 검사
                        if p[-1].retrieval_date >= newplate.retrieval_date:
                            # 자신보다 나중에 출고되는 강재가 있는데 새로운 pile 위에 올렸다면
                            penalty -= 10.0

            """ 만약에 penalty를 받지 않았다면 그 판단이 최선이었다는 것이므로 큰 보상을 줌"""
            if penalty == 0.0:
                return 10
            else: # penalty가 있다면
                if self.reward_heuristic in [1, 2, 3, 4]:
                    return 0 # 고정 penalty
                else:
                    return penalty # 다른 선택지의 수에 비례해서 더 증가하는 penalty

        """ 2. 만약 이미 쌓아 올린 pile에 추가했다면, 시간 간섭이 존재하지 않는가? """
        for i, plate in enumerate(pile[:-1]):# 마지막으로 적치된 강재(맨 위의 강재)는 리워드 계산 대상에 포함하지 않음
            move = 0
            if i + 1 + max_move >= len(pile):
                break
            for upper_plate in pile[i + 1:]:
                if plate.retrieval_date < upper_plate.retrieval_date:  # 하단의 강재 적치 기간이 짧은 경우 예상 크레인 횟수 증가
                    move += 1 # 강재 입장에서, 내가 출고되기 전까지 내 위에 있는 plate는 몇 개가 남겠는가?

            if move > max_move:  # 파일 내의 강재들 중 반출 시 예상 크레인 사용 횟수가 최대인 강재를 기준으로 보상 계산
                max_move = move
            total_move += move


        if max_move != 0:
            # reward = 1 / max_move  # 예상 크레인 사용 횟수의 역수로 보상 계산
            # reward = 10 / total_move
            # reward = - max_move
            # reward = - total_move

            if self.reward_heuristic in [1,3,5]:
                return 10 / max_move
            elif self.reward_heuristic in [2,4,6]:
                return 10 / total_move
            elif self.reward_heuristic == 7:
                return penalty - max_move
            else:
                return penalty - total_move
        else:
            reward = 2

        return reward
    def show_state(self, state):

        # 주어진 데이터
        data = np.flipud(state.cpu())

        # 첫 번째 열을 분리하여 새 열 추가
        # new_col = data[:, 0]
        data = np.insert(data, 1, -1, axis=1)
        # data[:, 0] = -1
        # data[:, 1] = new_col

        # 기본값 -1은 검은색으로, 양수는 그에 맞게 밝기가 감소하도록 표시
        # norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))
        norm = mcolors.Normalize(vmin=-1, vmax=100)
        colors = plt.cm.gray(norm(data))
        colors[data > 0] = plt.cm.Greys(norm(data[data > 0]))

        # pcolor를 사용하여 행렬 시각화
        plt.pcolor(data, cmap='gray', edgecolors='k', linewidths=1)

        # 축 설정
        plt.xlim(0, data.shape[1])
        plt.ylim(0, data.shape[0])

        # 저장할 이미지 파일 생성
        # buffer = BytesIO()
        # plt.savefig(buffer, format='png')
        # buffer.seek(0)

        # 이미지 파일 저장
        # 저장 경로 및 파일명을 적절히 변경하세요.
        # with open('output_image.png', 'wb') as f:
        #     f.write(buffer.read())

        # 이미지 표시 (선택사항)
        # plt.show()
        # pycharm error 429 : 28개 이상의 plot을 생성하면 문제가 됨
        buffer = BytesIO()
        plt.savefig(buffer, dpi = 72, format='png')
        buffer.seek(0)# 파일 포인터를 처음으로 되돌림

        return buffer


    def _get_state(self):
        state = np.full([self.max_height, self.num_piles + 1], -1)

        # 오늘 도착한 강재의 목록 중 max_height만큼만 자르기
        inbound_plates = [plate for plate in self.plates[:self.max_height] if plate.arrival_date == self.current_date]

        # 오늘 도착한 강재의 역순 배열 + 적치된 강재
        target_plates = [inbound_plates[::-1]] + self.piles[:] # self.piles[:]는 2차원 list

        for i, pile in enumerate(target_plates):
            for j, plate in enumerate(pile):
                state[j, i] = plate.retrieval_date - self.current_date

        # 오늘 도착한 강재와 현재 pile에 적치된 강재의 (투입 날짜) - (현재 날짜) 차이
        state = np.flipud(state)
        state = torch.tensor(state.copy()).to(self.device)

        return state

    def _retrieve_plates(self):
        for pile in self.piles:
            plates_retrieved = []

            for i, plate in enumerate(pile):
                if plate.retrieval_date <= self.current_date:
                    plates_retrieved.append(i)

            if len(plates_retrieved) > 0:
                self.crane_move += (len(pile) - plates_retrieved[0] - len(plates_retrieved))

            for index in plates_retrieved[::-1]:
                del pile[index]

    def _retrieve_all_plates(self):
        while True:
            # sum(list, []) : list bracket 2개를 1개로 줄여주는 역할
            next_retrieval_date = min(sum(self.piles, []), key=lambda x: x.retrieval_date).retrieval_date

            if next_retrieval_date != self.current_date:
                self.current_date = next_retrieval_date
                self._retrieve_plates()

            if not sum(self.piles, []):
                break


if __name__ == "__main__":
    from data import DataGenerator
    from GUI import GUI
    data_src = DataGenerator()
    num_piles = 10
    max_height = 20

    env = Stacking(data_src, num_piles, max_height)
    s = env.reset()
    images = []
    while True:
        a = random.choice([i for i in range(num_piles)])
        s_next, r, done = env.step(a)
        images.append(env.show_state(s))
        s = s_next

        if done:
            gui = GUI(images)
            break



