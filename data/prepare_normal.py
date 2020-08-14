import tensorflow as tf
import os
import random
import shutil
from embedded_model.object_detection.utils import dataset_util
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('output_path', '.', 'Path to output TFRecord')
flags.DEFINE_string('input_image_path', '.', 'Path to input images')
flags.DEFINE_string('input_label_path', '.', 'Path to input labels')
FLAGS = flags.FLAGS


def create_tf_example(image_name,img_path,label_path, label_tree=None):
    assert label_tree == None or isinstance(label_tree,dict)
    # TODO(user): Populate the following variables from your example.
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    image = Image.open(img_path)
    height = image.size[1] # Image height
    width = image.size[0] # Image width
    filename = image_name.encode() # Filename of the image. Empty if image is not from file
    encoded_image_data = encoded_jpg # Encoded image bytes
    image_format = b'jpeg' # b'jpeg' or b'png'

    f = open(label_path)
    d = f.read().split('\n')
    object_num = int(d[0])
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)
    if label_tree != None:
        unique_classes = label_tree.keys()
    else:
        unique_classes = []
    for p in range(object_num):
        if p >= 0:
            bndbox = []
            x = d[p + 1].split(' ')
            xmin = (int(x[0]) - 1) / width
            ymin = (int(x[1]) - 1) / height
            xmax = (int(x[2]) - 1) / width
            ymax = (int(x[3]) - 1) / height
            if xmin <= 0:
                xmin = 0
            if ymin <= 0:
                ymin = 0
            if xmax >= 1:
                xmax = 1.0
            if ymax >= 1:
                ymax = 1.0
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)
            if label_tree != None:
                # check if x[4] belongs to label_tree
                target = x[4].strip()
                for i, branch in enumerate(label_tree.keys()):
                    nodes = label_tree[branch]
                    if target not in nodes:
                        continue
                    classes_text.append(branch)
                    classes.append(i + 1)  # Zero for background
            else:
                classes_text.append(x[4])
                if x[4] not in unique_classes and x[4].strip() != "":
                    unique_classes.append(x[4])
                    index = len(unique_classes) - 1
                else:
                    index = unique_classes.index(x[4])
                classes.append(index)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def copy_to_train(input_image_dir, input_label_dir, output_dir, r=0.8):
    files = os.listdir(input_image_dir)
    num = len(files)
    sampled_num = int(num * 0.8)
    indexes = list(range(num))
    sampled = random.sample(indexes, sampled_num)
    rest = set(indexes) - set(sampled)
    train_path = os.path.join(output_dir, "train")
    train_image_path = os.path.join(train_path, "images")
    train_label_path = os.path.join(train_path, "labels")
    val_path = os.path.join(output_dir, "val")
    val_image_path = os.path.join(val_path, "images")
    val_label_path = os.path.join(val_path, "labels")

    os.makedirs(train_image_path)
    os.makedirs(val_image_path)
    os.makedirs(train_label_path)
    os.makedirs(val_label_path)

    for t in sampled:
        shutil.copy(os.path.join(input_image_dir, files[t]), train_image_path)
        shutil.copy(os.path.join(input_label_dir, '%s.txt' % os.path.splitext(files[t])[0]), train_label_path)

    for v in rest:
        shutil.copy(os.path.join(input_image_dir, files[v]), val_image_path)
        shutil.copy(os.path.join(input_label_dir, '%s.txt' % os.path.splitext(files[v])[0]), val_label_path)

    return train_image_path, train_label_path, val_image_path, val_label_path

def main(_):
    assert(FLAGS.output_path != '/')
    assert(FLAGS.output_path != '.')
    assert(FLAGS.output_path != '..')
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    train_image_path, train_label_path, val_image_path, val_label_path = copy_to_train(FLAGS.input_image_path, FLAGS.input_label_path, FLAGS.output_path)

    train_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, "train.record"))
    images = os.listdir(train_image_path)
    labels = list(range(len(images)))
    # TODO(user): Write code to read in your dataset to examples variable

    for i in range(len(images)):
        labels[i] = images[i].replace('jpg','txt')
        tf_example = create_tf_example(images[i],os.path.join(train_image_path,images[i]),os.path.join(train_label_path,labels[i]))
        train_writer.write(tf_example.SerializeToString())

    train_writer.close()


    val_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, "val.record"))
    images = os.listdir(val_image_path)
    labels = list(range(len(images)))
    # TODO(user): Write code to read in your dataset to examples variable

    label_tree = {
        'bucket0': ['bucket0','bucket01','bucket03'],
        'bucket1': ['bucket1', 'bucket19'],
        'Truck': ['Truck'],
    }
    for i in range(len(images)):
        labels[i] = images[i].replace('jpg','txt')
        tf_example = create_tf_example(images[i],os.path.join(val_image_path,images[i]),os.path.join(val_label_path,labels[i]), label_tree)
        val_writer.write(tf_example.SerializeToString())

    val_writer.close()


if __name__ == '__main__':
    tf.app.run()
