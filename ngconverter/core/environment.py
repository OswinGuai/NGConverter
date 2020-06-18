from enum import Enum

class Resource:
    class ResourceType(Enum):
        CPU = 0
        TF_GPU = 1

    required_num = 1

    def __init__(self, type: ResourceType, required_num: int):
        self.required_num = required_num
        self.type = type
