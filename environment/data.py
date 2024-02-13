import random
import numpy as np
import pandas as pd
import scipy.stats as stats


class DataGenerator:
    def __init__(self, num_plates=250):
        self.num_plates = num_plates  # 입고되는 강재의 평균 갯수

    def generate(self, file_path=None):
        # 실제 문제에 포함되는 강재의 수 샘플링
        num_plates = random.randint(int(0.9 * self.num_plates), int(1.1 * self.num_plates))

        # 강재 도착 간격 및 적치 기간에 대한 데이터 생성
        inter_arrival_time = np.floor(stats.expon.rvs(loc=0.0, scale=0.273, size=num_plates))
        stock_time = np.floor(stats.beta.rvs(1.85, 32783.4, loc=2.52, scale=738938.8, size=num_plates))

        # 강재의 입고일 및 출고일에 대한 데이터 생성
        plate_ids = np.arange(num_plates)
        arrival_dates = np.cumsum(inter_arrival_time)
        retrieval_dates = arrival_dates + stock_time

        df = pd.DataFrame({"plate id": plate_ids, "arrival date": arrival_dates, "retrieval date": retrieval_dates})

        if file_path is not None:
            df.to_excel(file_path, index=False)

        return df


if __name__ == "__main__":
    file_path = "./sample.xlsx"
    data_src = DataGenerator()
    data_src.generate(file_path)



