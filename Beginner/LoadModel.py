import pathlib
import torch
from BuildingModels import LinearRegressionModel
class LoadModels:
    def __init__(self, model_path='./model/01_MODEL_SAVE.pth'):
        loaded_model = LinearRegressionModel()
        loaded_model.state_dict(torch.load(f = model_path))

