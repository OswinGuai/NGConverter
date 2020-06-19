import os
import logging
from logging import Logger
from types import MethodType
from multiprocessing import Process

from ngconverter.record.exception import REPULICATE_TASK_NAME
from ngconverter.record.log import TaskLogger
from ngconverter.record.message import TaskMessage
from ngconverter.util.filesystem import remakedirs
from ngconverter.util.configparser import instance_embedded_model_config
from ngconverter.core.configuration import ConfigInfo
from ngconverter.core.constants import *
from ngconverter.core.finetune import FineTuneAPI
from ngconverter.core.convert import ConvertAPI
from ngconverter.core.environment import Resource

'''
    Task Management.
    author          - peizhyi@gmail.com
    last modified   - 20200619 16:58
'''
class Task:
    class Builder:
        @staticmethod
        def load_from_dir(task_dir):
            task = Task(task_dir, reloading=True)
            return task

        @staticmethod
        def init_by_config(task_name, config):
            task = Task(task_name, reloading=False)
            task_logger = task.get_tasklogger()
            assert isinstance(config, ConfigInfo)
            job = config.JOB
            task_func = config.FUNCTION
            task_pretrained = config.PRETRAINED_MODEL
            train_dataset_path = config.TRAIN_DATASET
            eval_dataset_path = config.EVAL_DATASET
            label_path = config.LABELSET
            train_steps = config.TRAIN_STEPS
            target_process = None

            def finetune_and_convert_process(self):
                assert isinstance(self, Task)
                finetuner = FineTuneAPI()
                converter = ConvertAPI()
                task_logger = self.get_tasklogger()
                if (task_func == ConfigInfo.FunctionType.OBJECT_DETECTION):
                    if (task_pretrained == ConfigInfo.EMBEDDED_MODEL):
                        pipeline_config_path = instance_embedded_model_config(EMBEDDED_SSD_PIPELINE_CONFIG_PATH,
                                                                              self.task_dir,
                                                                              train_dataset_path,
                                                                              eval_dataset_path,
                                                                              label_path)
                        self.status = TaskStatus.FINETUNING
                        task_logger.log(self, "Begin to fine-tune model by config file %s." % pipeline_config_path)
                        model_path = finetuner.finetune_embedded_objectdetection_model(pipeline_config_path, self.train_dir,
                                                                       train_steps=train_steps)
                        self.status = TaskStatus.FINETUNE_DONE
                        task_logger.log(self, "Finish fine-tuning.")

                        self.status = TaskStatus.CONVERTING
                        task_logger.log(self, "Begin to convert model at %s by config file %s." % (model_path, pipeline_config_path))
                        converter.convert_objectdetection_tf1(pipeline_config_path, model_path, self.task_dir)
                        self.status = TaskStatus.CONVERT_DONE
                        task_logger.log(self, "Finish converting.")
                    else:
                        raise NotImplementedError

                elif (task_func == ConfigInfo.FunctionType.IMAGE_CLASSIFICATION):
                    raise NotImplementedError
                else:
                    raise NotImplementedError

            if (job == ConfigInfo.JobType.FINETUNE_AND_CONVERT):
                target_process = finetune_and_convert_process
            else:
                raise NotImplementedError

            task.resource_list = [Resource(Resource.ResourceType.TF_GPU, 1)]
            task._process = MethodType(target_process, task)
            return task

    def __init__(self, task_name, parent_dir='.', reloading=True, process=None, level=logging.INFO):
        self.status = TaskStatus.INITING
        self.task_dir = os.path.join(parent_dir, task_name)
        self.task_name = task_name
        self._logger = TaskLogger.Factory.getTaskLogger(self, level)

        # Check directory empty
        # Or, reload the task with reloading option
        if os.path.exists(self.task_dir):
            if not reloading:
                self.status = TaskStatus.INIT_FAILED
                self._logger.error(self, TaskMessage.fill_in_template(TaskMessage.Template.error_create_duplicate_task, self))
                raise REPULICATE_TASK_NAME
            else:
                # TODO reload task.
                raise NotImplementedError
        # Build directories
        remakedirs(self.task_dir)
        self.train_dir = os.path.join(self.task_dir, "trained_model")

        self.status = TaskStatus.INITED
        self.resource_list = []
        self._process = process

    def prepare_resource(self):
        # Prepare GPU for training.
        for r in self.resource_list:
            assert isinstance(r, Resource)
            if r.resource_type == Resource.ResourceType.TF_GPU:
                #TODO check memory usage currently and choose a free one.
                assert r.required_num > 0
                os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(r.required_num)))

    def _process(self):
        '''
        Rewrite by builder.
        :return:
        '''
        raise NotImplementedError

    def execute(self):
        self.prepare_resource()
        self.p = Process(target=self._process)
        self.p.start()

    def hold(self):
        if self.p != None:
            self.p.join()

    def get_tasklogger(self) -> TaskLogger:
        return self._logger