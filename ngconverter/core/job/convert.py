from ngconverter.util import filesystem
from embedded_model.object_detection.protos import pipeline_pb2
import embedded_model.object_detection.export_tflite_ssd_graph_lib as ssd_exporter

from google.protobuf import text_format
import tensorflow as tf

import os


class ConvertAPI:

    def convert_objectdetection_tf2(self, pipeline_config_path, model_path, target_dir):
        # 1. Get inference graph.
        # TODO support from TF2 is still missing.

        # 2. Convert to tflite.
        converter = tf.lite.TFLiteConverter.from_saved_model("")
        tflite_model = converter.convert()
        raise NotImplementedError("support from TF2 is still missing.")

    def convert_imageclassification_tfhub_model(self, tfhub_model, target_dir):
        tfhub_model.export(target_dir)

    def convert_objectdetection_tf1(self, pipeline_config_path, model_path, target_dir):
        inference_graph_path = os.path.join(target_dir, "inference_graph")
        filesystem.remakedirs(inference_graph_path)
        input_file_path = os.path.join(inference_graph_path, "tflite_graph.pb")
        tflite_model_dir = os.path.join(target_dir, "tflite_model")
        filesystem.remakedirs(tflite_model_dir)
        tflite_model_path = os.path.join(tflite_model_dir, "detect.tflite")

        # 1. Get inference graph.
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.gfile.GFile(pipeline_config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        ssd_exporter.export_tflite_graph(pipeline_config,
                model_path,
                inference_graph_path,
                add_postprocessing_op=True,
                max_detections=10,
                max_classes_per_detection=1
                )

        # 2. Convert to tflite.
        input_arrays = ['normalized_input_image_tensor']
        output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(input_file_path, input_arrays, output_arrays, input_shapes={'normalized_input_image_tensor':[1,300,300,3]})
        converter.allow_custom_ops = True
        tflite_model = converter.convert()
        open(tflite_model_path, "wb").write(tflite_model)

