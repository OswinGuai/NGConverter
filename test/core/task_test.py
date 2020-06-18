import unittest

from ngconverter.core.task import Task
from ngconverter.util.configparser import load_config

"""
    Preliminary: You need to download test/resources from the repository to run this test.

"""
class TestTask(unittest.TestCase):



    def test_build_from_config(self):
        config_path = "test/resources/train_demo_config.yaml"
        config = load_config(config_path)
        task = Task.Builder.init_by_config(config)
        task.execute()

        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()