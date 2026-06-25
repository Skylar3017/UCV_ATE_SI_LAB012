from src.perceptron import Perceptron
from src.dataset import training_data


def test_perceptron_learns_positive_case():
    model = Perceptron()
    model.train(training_data)

    assert model.predict([9, 1]) == 1


def test_perceptron_learns_negative_case():
    model = Perceptron()
    model.train(training_data)

    assert model.predict([1, 0]) == 0