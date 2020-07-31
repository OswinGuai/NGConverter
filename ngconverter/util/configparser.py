import yaml
import os
from ngconverter.core.configuration import ConfigInfo
from embedded_model.object_detection.utils import label_map_util
from urllib import request
import shutil
import sys
from ngconverter.util.filesystem import try_makedirs


_EMBEDDED_MODEL_CHECKPOINT = "~/.nglite/pretrained"
_EMBEDDED_OBJECTDETECTION_NAME = "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03"
_EMBEDDED_OBJECTDETECTION_URL = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz"
def load_config(file_path):
    f = open(file_path)
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    config_info = ConfigInfo(yaml_config)
    return config_info

def instance_embedded_tf_objectdetection_model_config(
        embedded_model_config_path,
        target_dir,
        train_dataset_path,
        eval_dataset_path,
        label_path,
        batch_size=128,
        num_steps=10000):
    label_dict = label_map_util.get_label_map_dict(label_path, False)
    num_class = len(label_dict.keys())
    pretrained_model_dir = os.path.join(_EMBEDDED_MODEL_CHECKPOINT, _EMBEDDED_OBJECTDETECTION_NAME)
    try_makedirs(_EMBEDDED_MODEL_CHECKPOINT)
    if not os.path.exists(pretrained_model_dir):# Download and extract the model from the internet.
        sys.stdout.write("Pretrained model has not been deployed. Do it now...\n")
        local_pack = os.path.join("~/.nglite", _EMBEDDED_OBJECTDETECTION_NAME + ".tar.gz")
        sys.stdout.write("Download pretrained model from %s to %s...\n" % (_EMBEDDED_OBJECTDETECTION_URL, local_pack))
        sys.stdout.flush()
        request.urlretrieve(_EMBEDDED_OBJECTDETECTION_URL, local_pack)
        sys.stdout.write("Extract pretrained model to %s...\n" % pretrained_model_dir)
        sys.stdout.flush()
        shutil.unpack_archive(local_pack, _EMBEDDED_MODEL_CHECKPOINT)
    pretrained_model = os.path.join(pretrained_model_dir, "model.ckpt")
    update_config = {
        "num_classes": num_class,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "train_dataset_path": "\"%s\"" % train_dataset_path,
        "eval_dataset_path": "\"%s\"" % eval_dataset_path,
        "label_path": "\"%s\"" % label_path,
        "fine_tune_checkpoint": "\"%s\"" % pretrained_model
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
