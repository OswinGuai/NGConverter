import unittest


from ngconverter.util.configparser import load_config

"""
    Preliminary: You need to download test/resources from the repository to run this test.

"""
class TestConfig(unittest.TestCase):


    def test_default_config(self):

        config_path = "test/resources/demo_config.yaml"
        config = load_config(config_path)

        print(config)
        print(config.__dict__)
        self.assertEqual("embedded_data", config.EMBEDDED_DATA)
        self.assertEqual("object_detection", config.FUNCTION)


if __name__ == '__main__':
    unittest.main()