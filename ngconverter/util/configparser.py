import yaml
import os
from embedded_model.object_detection.utils import config_util
from ngconverter.core.configuration import ConfigInfo


def load_config(file_path):
    f = open(file_path)
    yaml_config = yaml.load(f)
    config_info = ConfigInfo(yaml_config)
    return config_info

def instance_tf_objectdetection_model_config(embedded_model_config, target_dir, train_dataset_path, eval_dataset_path, label_path):
    update_config = {
        "train_dataset_path": train_dataset_path,
        "eval_dataset_path": eval_dataset_path,
        "label_path": label_path,
        "fine_tune_checkpoint": "embedded_model/pretrained/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/model.ckpt"
    }
    input_config_list = [
        """
        train_input_reader: {{
            tf_record_input_reader {{
              input_path: "{train_dataset_path}"
            }}
            label_map_path: "{label_path}"
          }}
        """.format(**update_config),
        """
          eval_input_reader: {{
            tf_record_input_reader {{
              input_path: "{eval_dataset_path}"
            }}
            label_map_path: "{label_path}"
            shuffle: false
            num_readers: 1
          }}
        """.format(**update_config)
    ]
    pipeline_config_path = os.path.join(target_dir, "pipeline.config")
    if os.path.exists(pipeline_config_path):
        os.remove(pipeline_config_path)

    # configs = config_util.get_configs_from_pipeline_file(embedded_model_config, config_override=input_config_list)
    configs = config_util.get_configs_from_pipeline_file(embedded_model_config, config_override=None)

    pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_proto, target_dir)
    return pipeline_config_path

