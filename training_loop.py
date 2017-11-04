import tensorflow as tf
from network import net
import json
from data_load_utils import *
import numpy as np

batch_size, height, width, nchannels = 10, 360, 480, 3
learning_rate=0.005

locx, locy = parsepaths(train_paths)
x_im, y_im = loadimages(locx, locy)

with open('model_parameters.json') as params:
    params_dict = json.load(params)

params_dict['num_features'] = 16

# Forward pass
# with tf.Session() as sess:
    # read_input, output = tensorreshape(x_im[0],y_im[0], batch_size, width, height, nchannels)
    # model=net(read_input, params_dict)
# Not working, because cropping within the model is still required to solve: 
#    ValueError: Dimension 2 in both shapes must be equal, but are 45 and 44 for 'skip2_up' (op: 'ConcatV2') with input shapes: [1,60,45,176], [1,60,44,16], [] and with computed input tensors: input[2] = <-1>.

# Training loop
with tf.Session() as sess:
    log_dir="./"
    # We load a batch and reshape to tensor
    xbatch, ybatch = nextbatch(x_im, y_im, batchnumber=0, batchsize=batch_size)
    xb, yb = tensorreshape(xbatch, ybatch, batch_size, width, height, nchannels)
    # Obtain the predition
    y_hat, _ = net(xb, params_dict)
    # We calculate the loss
    loss = tf.losses.mean_squared_error(labels=yb, predictions=y_hat)
    total_loss = slim.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = slim.learning.create_train_op(total_loss, optimizer) 
    final_loss= slim.learning.train(train_op, logdir=log_dir, number_of_steps=200, save_summaries_secs=10, log_every_n_steps=50)
    
print("Done. With final loss: %s"%final_loss)
