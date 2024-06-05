from abc import ABC

import numpy as np

from estimations import Var, Mean


def bootstrap_resample(data, num_samples):
    """
    Выполняет бутстреп генерацию ревыборок из исходной выборки.

    Параметры:
    - data: массив данных
    - num_samples: количество ревыборок

    Возвращает:
    - bootstrap_samples: список ревыборок из исходной выборки
    """

    # Создание списка для хранения ревыборок
    bootstrap_samples = []

    # Генерация ревыборок
    for _ in range(num_samples):
        # Генерация случайной выборки с возвращением
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Добавление ревыборки в список
        bootstrap_samples.append(bootstrap_sample)
    return bootstrap_samples


class Modelling(ABC):

    def __init__(self, estimations: list, truth_value: float, number_resamples: int):
        self.estimations = estimations
        self.number_resamples = number_resamples
        self.truth_value = truth_value

        # Здесь будут храниться выборки оценок
        self.estimations_sample = np.zeros((len(self.estimations), self.number_resamples), dtype=np.float64)

    # Метод, оценивающий квадрат смещения оценок
    def estimate_bias_sqr(self):
        return np.array([(Mean().estimate(self.estimations_sample[i, :]) - self.truth_value) ** 2 for i in
                         range(len(self.estimations))])

    # Метод, оценивающий дисперсию оценок
    def estimate_var(self):
        return np.array([Var().estimate(self.estimations_sample[i, :]) for i in range(len(self.estimations))])

    # Метод, оценивающий СКО оценок
    def estimate_mse(self):
        return self.estimate_bias_sqr() + self.estimate_var()

    def get_samples(self):
        return self.estimations_sample

    def run(self):
        for i in range(2):
            for j in range(len(self.estimations[i])):
                self.estimations_sample[i][j] = self.estimations[i][j]
