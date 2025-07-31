from autogluon.tabular import TabularPredictor
from pathlib import Path

def load_model(path: Path):
    return TabularPredictor.load(path)