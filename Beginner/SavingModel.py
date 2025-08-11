import pathlib
import torch

class SavingModels:
    def __init__(self,model1):
        MODEL_PATH = pathlib.Path("model")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        MODEL_NAME = '01_MODEL_SAVE.pth'
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

        print(f"Saving Model to : {MODEL_SAVE_PATH}")
         # Make sure 'model1' is defined before this line
        torch.save(obj=model1.state_dict(), f=MODEL_SAVE_PATH)
