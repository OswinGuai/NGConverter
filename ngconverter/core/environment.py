from enum import Enum

class Resource:
    class ResourceType(Enum):
        CPU = 0
        TF_GPU = 1

    def __init__(self, resource_type: ResourceType = ResourceType.CPU, required_num: int = 1):
        self.required_num = required_num
        self.resource_type = resource_type
