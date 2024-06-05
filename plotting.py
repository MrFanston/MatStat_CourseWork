import matplotlib.pyplot as plt
import numpy as np

from estimations import HodgesLehmannEstimator, SampleMean
from modelling import bootstrap_resample, Modelling
from random_number_generator import SimpleRandomNumberGenerator, OutlierGenerator
from random_variables import LaplaceRandomVariable, SmoothedRandomVariable

POINTS = 100


def print_statistics(modelling: Modelling):
    mses = modelling.estimate_mse()
    print("[Выборочное среднее, Ходжеса-Лемана]")
    print("СКО:", mses)
    print("Соотношение СКО:", mses[1] / mses[0])
    variances = modelling.estimate_var()
    print("Дисперсия: ", variances)
    variances = modelling.estimate_bias_sqr()
    print("Квадрат смещения: ", variances)
    print('\n')


def plot_samples(samples, bandwidth):
    i = 1
    for sample in samples:
        x_min = min(sample)
        x_max = max(sample)
        x = np.linspace(x_min, x_max, POINTS)
        srv = SmoothedRandomVariable(sample, bandwidth)
        y = np.vectorize(srv.pdf)(x)
        if i == 1:
            name = 'Выборочное среднее'
        else:
            name = 'Ходжеса-Лемана'
        plt.plot(x, y, 'C' + i.__str__(), label = name )
        plt.legend()
        i += 1
    plt.show()


def run_modelling(number_resample: int, location: float, resamples: list):
    hodges_sample = [HodgesLehmannEstimator().estimate(sample) for sample in resamples]
    mean_sample = [SampleMean().estimate(sample) for sample in resamples]

    modelling = Modelling([mean_sample, hodges_sample], location, number_resample)
    modelling.run()

    return modelling


def without_emissions(location: float, scale: float, n: int, number_resample: int):
    rv = LaplaceRandomVariable(location, scale)
    generator = SimpleRandomNumberGenerator(rv)

    sample = generator.get(n)

    resamples = bootstrap_resample(sample, number_resample)

    modelling = run_modelling(number_resample, location, resamples)

    print_statistics(modelling)

    samples = modelling.get_samples()

    bandwidth = 0.1

    plot_samples(samples, bandwidth)


def with_symmetrical_emissions(location: float, scale: float, N: int, number_resample: int):
    rv = LaplaceRandomVariable(location, scale)

    generator = SimpleRandomNumberGenerator(rv)

    sample = generator.get(N)

    sample = OutlierGenerator.add_tukey_outliers(sample)

    resamples = bootstrap_resample(sample, number_resample)

    modelling = run_modelling(number_resample, location, resamples)

    print_statistics(modelling)

    samples = modelling.get_samples()

    bandwidth = 0.1

    plot_samples(samples, bandwidth)


def with_asymmetrical_emissions(location: float, scale: float, N: int, number_resample: int):
    rv = LaplaceRandomVariable(location, scale)

    generator = SimpleRandomNumberGenerator(rv)

    sample = generator.get(N)

    sample = OutlierGenerator.add_tukey_outliers(sample, symmetrical=False)

    resamples = bootstrap_resample(sample, number_resample)

    modelling = run_modelling(number_resample, location, resamples)

    print_statistics(modelling)

    samples = modelling.get_samples()

    bandwidth = 0.1
    plot_samples(samples, bandwidth)
