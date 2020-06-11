
from enum import Enum

class ModelWrapper:

    class Type(Enum):
        TENSORFLOW_SLIM = 10
        PYTORCH_ = 20

    def __init__(self, model_type):
        if not isinstance(model_type, ModelWrapper.Type):
            raise Exception("Wrong type!")
        self.model_type = model_type

class SlimModelWrapper(ModelWrapper):
    def __init__(self):
        super(ModelWrapper.Type.TENSORFLOW_SLIM)

