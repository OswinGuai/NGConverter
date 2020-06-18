import os
from enum import Enum
from types import MethodType
from multiprocessing import Process

from ngconverter.core.exception import *
from ngconverter.util.filesystem import remakedirs
from ngconverter.util.configparser import instance_embedded_model_config
from ngconverter.core.configuration import ConfigInfo
from ngconverter.core.constants import *
from ngconverter.core.finetune import FineTuneAPI
from ngconverter.core.convert import ConvertAPI
from ngconverter.core.environment import Resource


class Task:
    class Builder:
        @staticmethod
        def load_from_dir(task_dir):
            task = Task(task_dir, reloading=True)

        @staticmethod
        def init_by_config(task_name, config):
            task = Task(task_name, reloading=False)
            assert isinstance(config, ConfigInfo)
            job = config.JOB
            task_func = config.FUNCTION
            task_pretrained = config.PRETRAINED_MODEL
            train_dataset_path = config.TRAIN_DATASET
            eval_dataset_path = config.EVAL_DATASET
            label_path = config.LABELSET
            train_steps = config.TRAIN_STEPS
            target_dir = task_name
            train_dir = os.path.join(target_dir, "trained_model")
            model_path = os.path.join(train_dir, "model.ckpt-%d" % train_steps)
            target_process = None

            def finetune_and_convert_process(self):
                finetuner = FineTuneAPI()
                converter = ConvertAPI()
                if (task_func == ConfigInfo.FunctionType.OBJECT_DETECTION):
                    if (task_pretrained == ConfigInfo.EMBEDDED_MODEL):
                        
                        pipeline_config_path = instance_embedded_model_config(EMBEDDED_SSD_PIPELINE_CONFIG_PATH,
                                                                              task_name,
                                                                              train_dataset_path,
                                                                              eval_dataset_path,
                                                                              label_path)
                        finetuner.finetune_embedded_objectdetection_model(pipeline_config_path, train_dir,
                                                                       train_steps=train_steps)
                        converter.convert_objectdetection_tf1(pipeline_config_path, model_path, target_dir)

                    else:
                        raise NotImplementedError

                elif (task_func == ConfigInfo.FunctionType.IMAGE_CLASSIFICATION):
                    pass

            if (job == ConfigInfo.JobType.FINETUNE_AND_CONVERT):
                target_process = finetune_and_convert_process
            else:
                raise NotImplementedError

            task.resource_list = [Resource(Resource.ResourceType.TF_GPU, 1)]
            task._process = MethodType(target_process, task)
            return task


    class Status(Enum):
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

    def __init__(self, task_name, reloading=True, process=None):
        self.task_dir = task_name
        # Check directory empty
        # Or, reload the task with reloading option
        if os.path.exists(self.task_dir):
            if not reloading:
                self.status = Task.Status.INIT_FAILED
                raise REPULICATE_TASK_NAME
            else:
                # TODO reload task.
                pass
        self.status = Task.Status.INITED
        remakedirs(self.task_dir)
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

