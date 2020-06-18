import unittest

from ngconverter.core.task import Task
from ngconverter.util.configparser import load_config
import shutil

"""
    Preliminary: You need to download test/resources from the repository to run this test.

"""
class TestTask(unittest.TestCase):

    def tearDown(self):
        print("-----try to remove task.")
        shutil.rmtree(self.task_name)
        print("-----remove task finish.")

    def test_build_from_config(self):
        self.task_name = "finetune_convert_ssd_demotask"
        config_path = "test/resources/train_demo_config.yaml"
        config = load_config(config_path)
        task = Task.Builder.init_by_config(self.task_name, config)
        task.execute()
        print("-----waiting task.")
        task.hold()
        print("-----task finish.")

        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()
