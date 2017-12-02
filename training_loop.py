import os
import io as imp
import matplotlib.pyplot as plt
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
final_resized = 224
learning_rate = 0.001
model_version = 56

# Remap colours
_cmap = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (0, 0, 192), (128, 128, 0), (192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0), (0, 128, 192), (0, 0, 0)]
_mask_labels = {0: 'sky', 1: 'building', 2: 'column_pole', 3: 'road',
                    4: 'sidewalk', 5: 'tree', 6: 'sign', 7: 'fence', 8: 'car',
                    9: 'pedestrian', 10: 'byciclist', 11: 'void'}

def colorize(grayimg):
    gray_int=tf.cast(grayimg, tf.int32) # Cast to integer to allow to function as index
    gray_remapped=tf.gather(_cmap, gray_int) # Map gray to rgb
    gray_sq = tf.squeeze(gray_remapped) # Remove the dimension of 1
    return tf.cast(gray_sq, tf.float32) # Return as a float

# Legend
def legend():
    blocksize=0.1
    plt.figure(figsize=(5, 3))
    for i in range(len(_cmap)):
        xoffset=0
        yoffset=0
        if(i>5):
            xoffset=0.5
            yoffset=6*blocksize
        clr=(_cmap[i][0]/255,_cmap[i][1]/255,_cmap[i][2]/255)
        square = plt.Rectangle((0+xoffset, i*0.1-yoffset), width=blocksize, height=blocksize, fc=clr)
        plt.text(0.2+xoffset, 0.03+i*0.1-yoffset, _mask_labels[i], fontsize=10)
        plt.gca().add_patch(square)
    plt.ylim([0,0.6])
    plt.axis('off')
    buf = imp.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

buffer = legend()
# To TF image
legend = tf.image.decode_png(buffer.getvalue(), channels=4)
legend = tf.expand_dims(legend, 0)



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

    masked_weights = 1 - tf.unstack(one_hot_labels, axis=-1)[-1]

    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels,
        weights=masked_weights)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    if (image_in_tensorboard):
        yb = tf.cast(tf.divide(ground_truth_batch, 11), tf.float32)
        tf.summary.image("y", yb, max_outputs=1)

        predim = tf.nn.softmax(predictions)
        predimmax = tf.expand_dims(
            tf.cast(tf.argmax(predim, axis=3), tf.float32), -1)
        predimmaxdiv = tf.divide(tf.cast(predimmax, tf.float32), 11)
        tf.summary.image("y_hat", predimmaxdiv, max_outputs=1)

        tf.summary.image("Mask", tf.expand_dims(masked_weights, axis=-1), max_outputs=1)

        ediff = tf.minimum(tf.abs(tf.subtract(yb, predimmaxdiv)), tf.expand_dims(masked_weights, axis=-1))
        tf.summary.image("Error_difference", ediff, max_outputs=1)

        legend_sum = tf.summary.image("Legend", legend)

        tf.summary.image("Cy", colorize(ground_truth_batch), max_outputs=1)
        tf.summary.image("Cy_hat", colorize(predimmax), max_outputs=1)

        tf.summary.image("x", input_batch, max_outputs=1)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = slim.learning.create_train_op(
        total_loss, optimizer, summarize_gradients=False)
    print("Number of trainable parameters", np.sum(
        [np.prod(v.shape) for v in tf.trainable_variables()]))
    final_loss = slim.learning.train(
        train_op, logdir=log_dir,
        save_interval_secs=300,
        save_summaries_secs=30,
        log_every_n_steps=100)

print("Done. With final loss: %s" % final_loss)
