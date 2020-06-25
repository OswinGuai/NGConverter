import unittest

import os
import shutil

from ngconverter.core.finetune import FineTuneAPI
from ngconverter.core.convert import ConvertAPI
import embedded_model.object_detection.utils.config_util as config_util

"""
    Preliminary: You need to download test/resources from the repository to run this test.

"""
class TestTianyuanProcess(unittest.TestCase):

    finetuner = FineTuneAPI()
    converter = ConvertAPI()

    def test_objectdetection_wholeprocess(self):
        ori_pipeline_config_path = "embedded_model/ssd_pipeline.config"
        target_dir = "test/resources/demo_task"


        update_config = {
            "train_dataset_path": "test/resources/train.record",
            "eval_dataset_path": "test/resources/val.record",
            "label_path": "test/resources/ty.pbtxt"
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
        train_dir = os.path.join(target_dir, "trained_models")
        train_steps = 2
        model_path = os.path.join(train_dir, "model.ckpt-%d" % train_steps)

        configs = config_util.get_configs_from_pipeline_file(ori_pipeline_config_path, config_override=input_config_list)
        pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
        os.remove(pipeline_config_path)
        config_util.save_pipeline_config(pipeline_proto, target_dir)
        shutil.rmtree(train_dir)

        self.finetuner.finetune_embedded_objectdetection_model(pipeline_config_path, train_dir, train_steps=train_steps)
        print("finetune is finished.")

        self.converter.convert_objectdetection_tf1(pipeline_config_path, model_path, target_dir)
        print("convert is finished.")

        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()