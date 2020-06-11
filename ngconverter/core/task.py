
from enum import Enum

class Task:
    class Status(Enum):
        INITED = 0

        FINETUNING = 10
        FINETUNE_FINISH = 11
        FINETUNE_FAILED = 12

        CONVERTING = 20
        CONVERT_FINISH = 21
        CONVERT_FAILED = 22

        ALL_DONE = 100

    def __init__(self, task_dir, reloading=True):
        self.task_dir = task_dir
        # TODO check directory empty
        # TODO Or, reload the task with reloading option
        self.status = Status.INITED

