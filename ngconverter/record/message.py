from enum import Enum
from ngconverter.core.task import Task
class TaskMessage:

    class Template(Enum):
        error_create_duplicate_task = "Try to create the task at {task_dir} again, but reject."


    @staticmethod
    def fill_in_template(template: Template, task: Task):
        msg = template.value.format(**task.__dict__)
        return msg

