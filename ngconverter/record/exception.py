
class NGException(Exception):
    def __init__(self, id, msg):
        super().__init__()
        self.id = id
        self.msg = msg

#Task
REPULICATE_TASK_NAME = NGException(1000, "Task name exists. ")