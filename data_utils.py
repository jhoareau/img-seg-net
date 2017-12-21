# Largely based on the guides from
# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
# https://kwotsin.github.io/tech/2017/01/29/tfrecords.html

# Saving a dataset as tfrec can be done by:
#     tfrec_dump(paths, "filename.tfrec")
#         paths : either test_paths, train_paths, or valid_paths
#

# A batch can be loaded as follows:
#     tfsdataset = slim_dataset("filename.tfrec", num_samples)
#     images, _, annotations, _ = batch(tfsdataset, batch_size=3, height=360, width=480, resized=224)
#

from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import os
import random
slim = tf.contrib.slim

# Functions to store images, integers and strings in Tensorrec format


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


reader = tf.TFRecordReader

original_images = []

path_separator = "/"  # How your OS separates paths
data_root = "../CamVid"  # Where you unzipped the CamVid dataset
test_paths = data_root + path_separator + "test.txt"
train_paths = data_root + path_separator + "train.txt"
valid_paths = data_root + path_separator + "val.txt"


def parsepaths(location):
    # Load image and notations in separate lists
    x = []
    y = []
    n = []
    with open(location) as f:
        data = f.read()
    f.closed
    for l in data.split("\n"):
        if len(l) > 17:  # If the line contains more characters than the typical filename of one sample image
            x.append(path_replace(l.split(" ")[0]))
            y.append(path_replace(l.split(" ")[1]))
            n.append(l.split(" ")[0].replace("/SegNet/CamVid/train/", ""))
    return zip(x, y, n)


def path_replace(path):
    path = path.replace("/SegNet/CamVid/", "../")
    path = path.replace("/", path_separator)
    path = path.replace("..", data_root)
    return path


def tfrec_dump(dataset_paths, save_path):  # Either test_paths, train_paths or valid_paths
    filename_pairs = parsepaths(dataset_paths)
    writer = tf.python_io.TFRecordWriter(save_path)
    for img_path, annotation_path, file_name in filename_pairs:
        img = tf.gfile.FastGFile(img_path, 'rb').read()
        annotation = tf.gfile.FastGFile(annotation_path, 'rb').read()
        imgarr = np.array(Image.open(img_path))
        height = imgarr.shape[0]
        width = imgarr.shape[1]

        original_images.append((img, annotation))

        # Because the image is stored 1D we need to keep track of the image width and height

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/format': _bytes_feature(file_name[-3:].encode('ascii')),
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'file_name': _bytes_feature(file_name.encode('ascii')),
            'image/encoded': _bytes_feature(img),
            'annotation/encoded': _bytes_feature(annotation)}))  # We assume here that the other features of the annotation image are the same as for the photo image
        writer.write(example.SerializeToString())
    writer.close()

# Build the tfslim decoder and tfslim dataset


def slim_dataset(tfrec_location, num_samples):
    # How to interpret the dict keys
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'annotation/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    # How to decode certain keys
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'annotation': slim.tfexample_decoder.Image(image_key='annotation/encoded', format_key='image/format', channels=1),
    }

    items_to_descriptions = {
        'image': 'A 3-channel RGB coloured street image.',
        'annotation': 'A 1-channel image where everything is annotated.'
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    dataset = slim.dataset.Dataset(
        data_sources=tfrec_location,
        decoder=decoder,
        reader=reader,
        num_readers=4,
        num_samples=num_samples,
        items_to_descriptions=items_to_descriptions)

    return dataset

def random_flip_crop_image_and_labels(image, labels, feature_maps_image, feature_maps_annot, height, width):
    """Randomly crops `image` together with `labels`.
    Based on <https://stackoverflow.com/questions/42147427/tensorflow-how-to-randomly-crop-input-images-and-labels-in-the-same-way>

    Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
    Returns:
    A tuple of (cropped_image, cropped_label).
    """
    seed = random.randint(0, 1e10)
    combined = tf.concat([image, labels], axis=-1)

    last_label_dim = tf.shape(labels)[-1]
    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.random_crop(
        combined,
        size=[height, width, feature_maps_image + feature_maps_annot],
        seed=seed)
    combined_crop = tf.reshape(combined_crop, shape=(height, width, feature_maps_image + feature_maps_annot))
    maybe_flipped_images = tf.image.random_flip_left_right(combined_crop)
    crop_feature_maps = tf.unstack(maybe_flipped_images, axis=-1)
    return tf.stack(crop_feature_maps[:feature_maps_image], axis=-1), tf.stack(crop_feature_maps[feature_maps_image:], axis=-1)

# Convert to a tensor and resize
def imagepreprocessor(image, annot, height, width, scope=None):
    scopename="crop"
    with tf.name_scope(scope, scopename, [image, annot, height, width]):
        # First the image
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if annot.dtype != tf.float32:
            annot = tf.image.convert_image_dtype(annot, dtype=tf.float32)
        image, annot = random_flip_crop_image_and_labels(
                                image, annot,
                                feature_maps_image=3,
                                feature_maps_annot=1,
                                height=height,
                                width=width)
    return tf.expand_dims(image, 0), tf.expand_dims(tf.image.convert_image_dtype(annot, dtype=tf.uint8), 0)

# Load a batch


def batch(dataset, batch_size=3, height=360, width=480, resized=224):  # Resize to a multiple of 32
    IMAGE_HEIGHT = IMAGE_WIDTH = resized

    # First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=24 + 3 * batch_size,
        common_queue_min=24)

    # Get the images from provider
    raw_image, raw_annotation = data_provider.get(['image', 'annotation'])

    # Do image preprocessing
    image, annotation = imagepreprocessor(
        image=raw_image, annot=raw_annotation, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    # Reshape and batch
    '''raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    raw_annotation = tf.expand_dims(raw_annotation, 0)
    raw_annotation = tf.image.resize_nearest_neighbor(
        raw_annotation, [height, width])
    raw_annotation = tf.squeeze(raw_annotation)'''

    # Loaded batch
    images, annotations = tf.train.batch(
        [image, annotation],
        batch_size=batch_size,
        num_threads=4,
        capacity=4 * batch_size,
        allow_smaller_final_batch=True)
    return images, annotations
