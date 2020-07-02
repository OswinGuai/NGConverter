import unittest

from ngconverter.core.task import Task
from ngconverter.util.configparser import load_config
import time
import shutil

"""
    Preliminary: You need to download test/resources from the repository to run this test.

"""
class TestTask(unittest.TestCase):

    def tearDown(self):
        # shutil.rmtree(self.task_name)
        pass

    def test_build_from_config(self):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.task_name = "task_ty_classify_%s" % timestamp
        config_path = "tianyuan/ty_classification_config.yaml"
        config = load_config(config_path)
        task = Task.Builder.init_by_config(self.task_name, config)
        task.execute()
        task.hold()

if __name__ == '__main__':
    unittest.main()
