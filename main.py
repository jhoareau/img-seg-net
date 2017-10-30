import tensorflow as tf
from network import net
import json

with open('model_parameters.json') as params:
    params_dict = json.load(params)

params_dict['num_features'] = 16
params_dict['kernel_size'] = 3

with tf.Session() as sess:
    net(tf.placeholder(tf.float32, shape=(10, 320, 320, 3)), params_dict)
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter('train', sess.graph)
