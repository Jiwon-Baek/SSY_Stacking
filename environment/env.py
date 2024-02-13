import random
import numpy as np
import pandas as pd

from environment.data import DataGenerator


class Plate:
    def __init__(self, id, arrival_date, retrieval_date):
        self.id = id
        self.arrival_date = arrival_date
        self.retrieval_date = retrieval_date


class Stacking:
    def __init__(self, data_src, num_piles=4, max_height=4, device=None):
        self.data_src = data_src
        self.num_piles = num_piles
        self.max_height = max_height  # 한 파일에 적치 가능한 강재의 수
        self.device = device

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

        self.stage = 0
        self.current_date = 0
        self.crane_move = 0

    def step(self, action):
        done = False
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

        next_state = self._get_state()  # 쌓인 강재들 리스트에서 state 를 계산

        if done:
            self._retrieve_all_plates()

        return next_state, reward, done

    def reset(self):
        self.current_date = min(self.plates, key=lambda x: x.arrival_date).arrival_date
        self.crane_move = 0
        return self._get_state()

    def _calculate_reward(self, action):
        pile = self.piles[action]
        max_move = 0

        if len(pile) == 1:
            return 0

        for i, plate in enumerate(pile[:-1]):
            move = 0
            if i + 1 + max_move >= len(pile):
                break
            for upper_plate in pile[i + 1:]:
                if plate.retrieval_date < upper_plate.retrieval_date:  # 하단의 강재 적치 기간이 짧은 경우 예상 크레인 횟수 증가
                    move += 1
            if move > max_move:  # 파일 내의 강재들 중 반출 시 예상 크레인 사용 횟수가 최대인 강재를 기준으로 보상 계산
                max_move = move

        if max_move != 0:
            reward = 1 / max_move  # 예상 크레인 사용 횟수의 역수로 보상 계산
        else:
            reward = 2

        return reward

    def _get_state(self):
        state = np.full([self.max_height, self.num_piles + 1], -1)

        inbound_plates = [plate for plate in self.plates[:self.max_height] if plate.arrival_date == self.current_date]
        target_plates = [inbound_plates[::-1]] + self.piles[:]

        for i, pile in enumerate(target_plates):
            for j, plate in enumerate(pile):
                state[j, i] = plate.retrieval_date - self.current_date

        state = np.flipud(state)

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
            next_retrieval_date = min(sum(self.piles, []), key=lambda x: x.retrieval_date).retrieval_date

            if next_retrieval_date != self.current_date:
                self.current_date = next_retrieval_date
                self._retrieve_plates()

            if not sum(self.piles, []):
                break


if __name__ == "__main__":
    data_src = DataGenerator()
    num_piles = 10
    max_height = 20

    env = Stacking(data_src, num_piles, max_height)
    s = env.reset()

    while True:
        a = random.choice([i for i in range(num_piles)])
        s_next, r, done = env.step(a)
        print(r)
        s = s_next

        if done:
            break



