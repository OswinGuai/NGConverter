from __future__ import annotations

import os
import sys
import logging
from logging import Logger


class TaskLogger:
    def __init__(self, task, logger: Logger):
        self._logger = logger
        self._task = task

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, extra={"task_status": self._task.status}, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._logger.warn(msg, extra={"task_status": self._task.status}, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, extra={"task_status": self._task.status}, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._logger.exception(msg, extra={"task_status": self._task.status}, *args, **kwargs)

    @staticmethod
    def getTaskLogger(task, level=logging.INFO) -> TaskLogger:
        task_logger = logging.getLogger(task.task_name)
        handler = logging.FileHandler(os.path.join(task.task_dir, 'task.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(task_status)s - %(message)s')
        handler.setFormatter(formatter)
        task_logger.addHandler(handler)
        task_logger.setLevel(level)
        task_logger = TaskLogger(task, task_logger)
        return task_logger


