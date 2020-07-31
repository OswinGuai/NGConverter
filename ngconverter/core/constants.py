from enum import Enum
from pathlib import Path
home = str(Path.home())

EMBEDDED_SSD_PIPELINE_CONFIG_PATH = "%s/.nglite/empty_ssd_pipeline.config" % home

class TaskStatus(Enum):
    '''
    Status describing task.
    '''

    # Initialization
    INITING = 0
    INITED = 1
    INIT_FAILED = 2

    # Execution
    RESOURCE_PREPARING = 100

    FINETUNING = 110
    FINETUNE_DONE = 111
    FINETUNE_FAILED = 112

    CONVERTING = 120
    CONVERT_DONE = 121
    CONVERT_FAILED = 122

    CLEANING_TEMP_FILES = 130

    EXECUTION_DONE = 199

    # Response
    RESPONDING = 200
    RESPOND_DONE = 201
    RESPOND_FAILED = 202