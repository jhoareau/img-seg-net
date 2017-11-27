import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from network import net
import json
from data_utils import *
slim = tf.contrib.slim
# This might increase training time, so set to False if desired
image_in_tensorboard = True
batch_size, height, width, nchannels = 3, 360, 480, 3
# batch_size, height, width, nchannels = 3, 240, 320, 3
final_resized = 224
# final_resized = 192
learning_rate = 0.001
model_version = 56

with open('model_parameters.json') as params:
    params_dict = json.load(params)[repr(model_version)]

params_dict['input_num_features'] = 48
# params_dict['input_num_features'] = 16
params_dict['output_classes'] = 12

# Save training data in tfrec and load it into a slim dataset

tfrec_dump(train_paths, "trainset.tfrec")
tfsdataset = slim_dataset("trainset.tfrec", 367)

gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# Training loop
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
    log_dir = 'train'
    # We load a batch and reshape to tensor
    xbatch, ybatch = batch(
        tfsdataset, batch_size=batch_size, height=height, width=width, resized=final_resized)
    input_batch = tf.reshape(xbatch, shape=(
        batch_size, final_resized, final_resized, 3))
    ground_truth_batch = tf.reshape(ybatch, shape=(
        batch_size, final_resized, final_resized, 1))

    # Obtain the prediction
    predictions = net(input_batch, params_dict)

    # We calculate the loss
    one_hot_labels = slim.one_hot_encoding(
        tf.squeeze(ground_truth_batch),
        params_dict['output_classes'])

    '''class_weights = [1.0 for i in range(params_dict['output_classes'] - 1)]
    class_weights.append(0.0)
    masked_weights = tf.reduce_sum(tf.multiply(one_hot_labels, class_weights), 1)'''

    masked_weights = 1 - tf.unstack(one_hot_labels, axis=-1)[-1]

    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels,
        weights=masked_weights)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    if (image_in_tensorboard):
        yb = tf.cast(tf.divide(ground_truth_batch, 11), tf.float32)
        tf.summary.image("x", input_batch, max_outputs=1)
        tf.summary.image("y", yb, max_outputs=1)
        predim = tf.nn.softmax(predictions)
        predimmax = tf.expand_dims(
            tf.cast(tf.argmax(predim, axis=3), tf.float32), -1)
        predimmax = tf.divide(tf.cast(predimmax, tf.float32), 11)
        tf.summary.image("y_hat", predimmax, max_outputs=1)
        ediff = tf.abs(tf.subtract(yb, predimmax))
        tf.summary.image("Error difference", ediff, max_outputs=1)
        tf.summary.image("Mask", tf.expand_dims(masked_weights, axis=-1), max_outputs=1)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = slim.learning.create_train_op(
        total_loss, optimizer, summarize_gradients=False)
    print("Number of trainable parameters", np.sum(
        [np.prod(v.shape) for v in tf.trainable_variables()]))
    final_loss = slim.learning.train(
        train_op, logdir=log_dir, number_of_steps=10000, save_summaries_secs=10, log_every_n_steps=50)

print("Done. With final loss: %s" % final_loss)
