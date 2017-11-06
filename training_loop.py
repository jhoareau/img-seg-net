import tensorflow as tf
from network import net
import json
from data_load_utils import *
import numpy as np
slim = tf.contrib.slim

batch_size, height, width, nchannels = 10, 360, 480, 3
learning_rate = 0.001

locx, locy = parsepaths(train_paths)
x_im, y_im = loadimages(locx, locy)

with open('model_parameters.json') as params:
    params_dict = json.load(params)

params_dict['input_num_features'] = 48
params_dict['output_classes'] = 12
params_dict['num_features'] = 16


# Training loop
with tf.Session() as sess:
    log_dir = 'train'
    # We load a batch and reshape to tensor
    xbatch, ybatch = nextbatch(x_im, y_im, batchnumber=0, batchsize=batch_size)
    input_batch, ground_truth_batch = tensorreshape(xbatch, ybatch, batch_size,
                           width, height, nchannels)

    # Obtain the prediction
    predictions = net(input_batch, params_dict)

    # We calculate the loss
    one_hot_labels = slim.one_hot_encoding(
        ground_truth_batch,
        params_dict['output_classes'])
    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)
    final_loss = slim.learning.train(
        train_op, logdir=log_dir, number_of_steps=200, save_summaries_secs=10, log_every_n_steps=50)

print("Done. With final loss: %s" % final_loss)
