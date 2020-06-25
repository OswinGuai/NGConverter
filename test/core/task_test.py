import unittest

from ngconverter.core.task import Task
from ngconverter.util.configparser import load_config
import shutil

"""
    Preliminary: You need to download test/resources from the repository to run this test.

"""
class TestTask(unittest.TestCase):

    def tearDown(self):
        # shutil.rmtree(self.task_name)
        pass

    def test_build_from_config(self):
        self.task_name = "finetune_convert_imageclassification_demotask"
        config_path = "test/resources/train_imageclassification_demo_config.yaml"
        config = load_config(config_path)
        task = Task.Builder.init_by_config(self.task_name, config)
        task.execute()
        task.hold()

        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()
