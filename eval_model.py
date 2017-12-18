import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from network import net
import json
from data_utils import *
from colorise_camvid import colorize, legend, _mask_labels
from iou_calculation import intersection_over_union
slim = tf.contrib.slim

n_images = 2

batch_size, height, width, nchannels = n_images, 360, 480, 3
final_resized = 224
model_version = 56

with open('model_parameters.json') as params:
    params_dict = json.load(params)[repr(model_version)]

params_dict['input_num_features'] = 48
params_dict['output_classes'] = 12

tfrec_dump(test_paths, "testset.tfrec")
tfsdataset = slim_dataset("testset.tfrec", n_images)

gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# Training loop
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
    log_dir = 'test'
    # We load a batch and reshape to tensor
    xbatch, ybatch = batch(
        tfsdataset, batch_size=batch_size, height=height, width=width, resized=final_resized)
    input_batch = tf.reshape(xbatch, shape=(batch_size, final_resized, final_resized, 3))
    ground_truth_batch = tf.reshape(ybatch, shape=(batch_size, final_resized, final_resized, 1))

    # Obtain the prediction
    predictions = net(input_batch, params_dict, is_training=False)
    predim = tf.nn.softmax(predictions)
    predimmax = tf.expand_dims(
        tf.cast(tf.argmax(predim, axis=3), tf.float32), -1)

    yb = tf.cast(tf.divide(ground_truth_batch, 11), tf.float32)
    predimmaxdiv = tf.divide(tf.cast(predimmax, tf.float32), 11)

    # We calculate the loss
    one_hot_labels = slim.one_hot_encoding(
        tf.squeeze(ground_truth_batch),
        params_dict['output_classes'])

    masked_weights = 1 - tf.unstack(one_hot_labels, axis=-1)[-1]

    # Concat all four images into one grid
    ediff = tf.minimum(tf.abs(tf.subtract(yb, predimmaxdiv)), tf.expand_dims(masked_weights, axis=-1))
    norm_ediff = tf.ceil(ediff)
    annots=tf.concat([colorize(ground_truth_batch),colorize(predimmax)],2)
    img_and_err=tf.multiply(tf.concat([input_batch,tf.image.grayscale_to_rgb(norm_ediff)],2),255) # Multiply by 255, because it actually outputs 0.0 to 1.0
    aio=tf.concat([img_and_err,annots],1)
    tf.summary.image("All_in_one", aio, max_outputs=n_images)
    tf.summary.image("Legend", legend, max_outputs=1)

    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels,
        weights=masked_weights)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    accuracy = tf.reduce_mean(tf.cast(norm_ediff, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    iou_array, mean_iou = intersection_over_union(ground_truth_batch, predimmax, params_dict['output_classes'], masked_weights)
    tf.summary.scalar('mean_IoU', mean_iou)
    class_labels = tf.convert_to_tensor(np.array(list(_mask_labels.values())), tf.string)
    iou_per_class = tf.stack([class_labels, tf.as_string(iou_array, precision=2)], axis=1)
    tf.summary.text('IoU per class', iou_per_class)

    slim.evaluation.evaluate_once(
        '',
        'train_aws_nbs/model.ckpt-10963',
        'test'
    )
