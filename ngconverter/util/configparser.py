import yaml
import os
from ngconverter.core.configuration import ConfigInfo


def load_config(file_path):
    f = open(file_path)
    yaml_config = yaml.load(f)
    config_info = ConfigInfo(yaml_config)
    return config_info

def instance_tf_objectdetection_model_config(embedded_model_config_path, target_dir, train_dataset_path, eval_dataset_path, label_path):
    update_config = {
        "num_classes": 2,
        "batch_size": 128,
        "num_steps": 20000,
        "train_dataset_path": "\"%s\"" % train_dataset_path,
        "eval_dataset_path": "\"%s\"" % eval_dataset_path,
        "label_path": "\"%s\"" % label_path,
        "fine_tune_checkpoint": "\"%s\"" % "embedded_model/pretrained/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/model.ckpt"
    }
    with open(embedded_model_config_path) as embedded_file:
        content = embedded_file.read()
    filled_content = content.format(**update_config)
    pipeline_config_path = os.path.join(target_dir, "pipeline.config")
    if os.path.exists(pipeline_config_path):
        os.remove(pipeline_config_path)
    with open(pipeline_config_path, 'w') as filled_config:
        filled_config.write(filled_content)
    return pipeline_config_path
