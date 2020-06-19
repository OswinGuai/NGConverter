from __future__ import annotations

import os
import logging
from logging import Logger


class TaskLogger:
    def __init__(self, logger):
        assert isinstance(logger, Logger)
        self._logger = logger

    def info(self, task, msg, *args, **kwargs):
        self._logger.info(msg, extra={"task_status": task.status}, *args, **kwargs)

    def warn(self, task, msg, *args, **kwargs):
        self._logger.warn(msg, extra={"task_status": task.status}, *args, **kwargs)

    def error(self, task, msg, *args, **kwargs):
        self._logger.error(msg, extra={"task_status": task.status}, *args, **kwargs)

    def exception(self, task, msg, *args, **kwargs):
        self._logger.exception(msg, extra={"task_status": task.status}, *args, **kwargs)

    @staticmethod
    def getTaskLogger(task, level=logging.INFO) -> TaskLogger:
        task_logger = logging.getLogger(task.task_name)
        handler = logging.FileHandler(os.path.join(task.task_dir, 'task.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(task_status)s - %(message)s')
        handler.setFormatter(formatter)
        task_logger.addHandler(handler)
        task_logger.setLevel(level)
        task_logger.info("what!!!")
        task_logger = TaskLogger(task_logger)
        return task_logger
