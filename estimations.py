import statistics
from abc import ABC, abstractmethod
from statistics import median


class Estimation(ABC):
    @abstractmethod
    def estimate(self, sample):
        pass


class SampleMean(Estimation):
    """
    SampleMean (Выборочное среднее)
    """

    def estimate(self, sample):
        """
        Estimate вычисляет выборочное среднее значение для переданной выборки.
        """
        return sum(sample) / len(sample)


class HodgesLehmannEstimator(Estimation):
    def estimate(self, sample):
        """
        Estimate вычисляет оценку Ходжеса-Лемана на основе переданной выборки.
        """
        # Создаем список для хранения средних значений Уолша
        walsh_average = []

        # Вычисляем средние значения Уолша для всех пар элементов выборки
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                walsh_average.append((sample[i] + sample[j]) / 2)

        # Сортируем средние значения Уолша
        walsh_average.sort()

        # Возвращаем медиану средних значений Уолша
        return median(walsh_average)


class Mean(Estimation):
    def estimate(self, sample):
        return statistics.mean(sample)


class Var(Estimation):
    def estimate(self, sample):
        return statistics.variance(sample)

