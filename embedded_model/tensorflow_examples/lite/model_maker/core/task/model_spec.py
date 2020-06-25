# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import os
import re
import tempfile

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat
from tensorflow_examples.lite.model_maker.core import file_util
from tensorflow_examples.lite.model_maker.core.task import hub_loader

import tensorflow_hub as hub
from tensorflow_hub import registry

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def _get_compat_tf_versions(compat_tf_versions=None):
  """Gets compatible tf versions (default: [2]).

  Args:
    compat_tf_versions: int, int list or None, indicates compatible versions.

  Returns:
    A list of compatible tf versions.
  """
  if compat_tf_versions is None:
    compat_tf_versions = [2]
  if not isinstance(compat_tf_versions, list):
    compat_tf_versions = [compat_tf_versions]
  return compat_tf_versions


def get_num_gpus(num_gpus):
  try:
    tot_num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
  except TypeError:
    tf.compat.v1.logging.warning("Couldn't get the number of gpus.")
    tot_num_gpus = max(0, num_gpus)
  if num_gpus > tot_num_gpus or num_gpus == -1:
    num_gpus = tot_num_gpus
  return num_gpus


class ImageModelSpec(object):
  """A specification of image model."""

  mean_rgb = [0.0]
  stddev_rgb = [255.0]

  def __init__(self,
               uri,
               compat_tf_versions=None,
               input_image_shape=None,
               name=''):
    self.uri = uri
    self.compat_tf_versions = _get_compat_tf_versions(compat_tf_versions)
    self.name = name

    if input_image_shape is None:
      input_image_shape = [224, 224]
    self.input_image_shape = input_image_shape


mobilenet_v2_spec = ImageModelSpec(
    uri='https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/feature_vector/4',
    compat_tf_versions=2,
    name='mobilenet_v2')

resnet_50_spec = ImageModelSpec(
    uri='https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/feature_vector/4',
    compat_tf_versions=2,
    name='resnet_50')

efficientnet_lite0_spec = ImageModelSpec(
    uri='https://hub.tensorflow.google.cn/tensorflow/efficientnet/lite0/feature-vector/2',
    compat_tf_versions=[1, 2],
    name='efficientnet_lite0')

efficientnet_lite1_spec = ImageModelSpec(
    uri='https://hub.tensorflow.google.cn/tensorflow/efficientnet/lite1/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[240, 240],
    name='efficientnet_lite1')

efficientnet_lite2_spec = ImageModelSpec(
    uri='https://hub.tensorflow.google.cn/tensorflow/efficientnet/lite2/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[260, 260],
    name='efficientnet_lite2')

efficientnet_lite3_spec = ImageModelSpec(
    uri='https://hub.tensorflow.google.cn/tensorflow/efficientnet/lite3/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[280, 280],
    name='efficientnet_lite3')

efficientnet_lite4_spec = ImageModelSpec(
    uri='https://hub.tensorflow.google.cn/tensorflow/efficientnet/lite4/feature-vector/2',
    compat_tf_versions=[1, 2],
    input_image_shape=[300, 300],
    name='efficientnet_lite4')


# A dict for model specs to make it accessible by string key.
MODEL_SPECS = {
    'efficientnet_lite0': efficientnet_lite0_spec,
    'efficientnet_lite1': efficientnet_lite1_spec,
    'efficientnet_lite2': efficientnet_lite2_spec,
    'efficientnet_lite3': efficientnet_lite3_spec,
    'efficientnet_lite4': efficientnet_lite4_spec,
    'mobilenet_v2': mobilenet_v2_spec,
    'resnet_50': resnet_50_spec,
}

# List constants for supported models.
IMAGE_CLASSIFICATION_MODELS = [
    'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2',
    'efficientnet_lite3', 'efficientnet_lite4', 'mobilenet_v2', 'resnet_50'
]

def get(spec_or_str):
  """Gets model spec by name or instance, and initializes by default."""
  if isinstance(spec_or_str, str):
    model_spec = MODEL_SPECS[spec_or_str]
  else:
    model_spec = spec_or_str

  if inspect.isclass(model_spec):
    return model_spec()
  else:
    return model_spec
