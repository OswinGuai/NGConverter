from __future__ import annotations

import os
import logging
from logging import Logger


class TaskLogger:
    def __init__(self, logger):
        assert isinstance(logger, Logger)
        self._logger = logger

    def log(self, task, level, msg, *args, **kwargs):
        self._logger.log(level, msg, extra={"task_status": task.status}, *args, **kwargs)

    def warn(self, task, level, msg, *args, **kwargs):
        self._logger.warn(level, msg, extra={"task_status": task.status}, *args, **kwargs)

    def error(self, task, level, msg, *args, **kwargs):
        self._logger.error(level, msg, extra={"task_status": task.status}, *args, **kwargs)

    def exception(self, task, level, msg, *args, **kwargs):
        self._logger.exception(level, msg, extra={"task_status": task.status}, *args, **kwargs)

    @staticmethod
    def getTaskLogger(task, level=logging.INFO) -> TaskLogger:
        task_logger = logging.getLogger(task.task_name)
        handler = logging.FileHandler(os.path.join(task.task_dir, 'task.log'))
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(task_status) - %(message)s')
        handler.setFormatter(formatter)
        task_logger.addHandler(handler)
        return task_logger
