import tensorflow as tf
from network import net
import json

with open('model_parameters.json') as params:
    params_dict = json.load(params)

with tf.Session() as sess:
    net(tf.placeholder(tf.float32, shape=(1, 300, 300, 3)), params_dict)
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter('train', sess.graph)
