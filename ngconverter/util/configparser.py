import yaml
from ngconverter.core.configuration import ConfigInfo

def load_config(file_path):
    f = open(file_path)
    yaml_config = yaml.load(f)
    config_info = ConfigInfo(yaml_config)
    return config_info
