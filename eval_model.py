import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from network import net
import json
from data_utils import *
from colorise_camvid import colorize, legend
slim = tf.contrib.slim

n_images = 5

batch_size, height, width, nchannels = n_images, 360, 480, 3
final_resized = 224
model_version = 56

with open('model_parameters.json') as params:
    params_dict = json.load(params)[repr(model_version)]

params_dict['input_num_features'] = 48
params_dict['output_classes'] = 12

tfrec_dump(valid_paths, "validset.tfrec")
tfsdataset = slim_dataset("validset.tfrec", n_images)

gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# Training loop
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
    log_dir = 'val'
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
    predim = tf.nn.softmax(predictions)
    predimmax = tf.expand_dims(
        tf.cast(tf.argmax(predim, axis=3), tf.float32), -1)
    predimmaxdiv = tf.divide(tf.cast(predimmax, tf.float32), 11)

    tf.summary.image("x", input_batch, max_outputs=n_images)
    tf.summary.image("y", yb, max_outputs=n_images)
    tf.summary.image("y_hat", predimmaxdiv, max_outputs=n_images)
    tf.summary.image("Cy", colorize(ground_truth_batch), max_outputs=n_images)
    tf.summary.image("Cy_hat", colorize(predimmax), max_outputs=n_images)
    tf.summary.image("Legend", legend, max_outputs=1)

    # We calculate the loss
    one_hot_labels = slim.one_hot_encoding(
        tf.squeeze(ground_truth_batch),
        params_dict['output_classes'])

    masked_weights = 1 - tf.unstack(one_hot_labels, axis=-1)[-1]

    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels,
        weights=masked_weights)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    slim.evaluation.evaluate_once(
        '',
        'train_aws/model.ckpt-62424',
        'val'
    )
