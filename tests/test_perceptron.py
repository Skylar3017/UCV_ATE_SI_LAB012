from src.perceptron import Perceptron
from src.dataset import training_data
import subprocess
import sys

def test_perceptron_learns_positive_case():
    model = Perceptron()
    model.train(training_data)

    assert model.predict([9, 1]) == 1


def test_perceptron_learns_negative_case():
    model = Perceptron()
    model.train(training_data)

    assert model.predict([1, 0]) == 0

def test_main_executes_correctly():
    result = subprocess.run(
        [sys.executable, "-m", "src.main"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "1"