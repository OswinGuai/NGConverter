from enum import Enum

class ConfigInfo:

    EMBEDDED_MODEL = "embedded_model"
    EMBEDDED_DATA = "embedded_data"
    EMBEDDED_LABEL = "embedded_label"

    DEFAULT_TRAINSTEPS = 1000

    def __init__(self, yaml_config):

        self.TASK_TYPE = ConfigInfo.TaskType.UNDEFINED
        self.FUNCTION = ConfigInfo.FunctionType.UNDEFINED
        self.PRETRAINED_MODEL = ConfigInfo.EMBEDDED_MODEL
        self.TRAIN_DATASET = ConfigInfo.EMBEDDED_DATA
        self.EVAL_DATASET = ConfigInfo.EMBEDDED_DATA
        self.LABELSET = ConfigInfo.EMBEDDED_LABEL
        self.TARGET_PLATFORM = ConfigInfo.PlatformType.UNDEFINED
        self.TRAIN_STEPS = ConfigInfo.DEFAULT_TRAINSTEPS

        base_memberset = set(self.__dict__.keys())

        user_config = yaml_config
        user_config = _uppercase_for_dict_keys(user_config)
        self.__dict__.update(user_config)

        updated_memberset = set(self.__dict__.keys())
        assert base_memberset == updated_memberset


    class TaskType(Enum):
        UNDEFINED = "undefined"
        FINETUNE_AND_CONVERT = "finetune_and_convert"

    class FunctionType(Enum):
        UNDEFINED = "undefined"
        OBJECT_DETECTION = "object_detection"
        IMAGE_CLASSIFICATION = "image_classification"

    class PlatformType(Enum):
        UNDEFINED = "undefined"
        ANDROID = "android"
        IOS = "ios"


def _uppercase_for_dict_keys(lower_dict):
    upper_dict = {}
    for k, v in lower_dict.items():
        if isinstance(v, dict):
            v = _uppercase_for_dict_keys(v)
        upper_dict[k.upper()] = v
    return upper_dict