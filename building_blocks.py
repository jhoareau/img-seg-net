import tensorflow.contrib.slim as slim
import tensorflow as tf


def dense_block(x, num_layers, num_features, scope, p=0.2, skipped=True):
    layers = []
    for i in range(num_layers):
        layer = create_layer(x, num_features, p=p,
                             scope=(scope + "/layer" + str(i)))
        layers.append(layer)
        if (skipped == False and i == num_layers - 1):
            continue
        x = tf.concat(axis=-1, values=[x, layer],
                      name=(scope + "/skip" + str(i)))
    return x, layers


def create_layer(input, num_features, scope, kernel_size=3, p=0.2):
    relud_batch_norm = slim.batch_norm(
        input, activation_fn=tf.nn.relu, scope=(scope + "/batchnorm"))
    conv = slim.conv2d(relud_batch_norm, num_features,
                       kernel_size, scope=(scope + "/conv"), activation_fn=None)
    dropout = slim.dropout(conv, p, scope=(scope + "/dropout"))
    return dropout


def transition_down(input, num_features, scope, kernel_size=1, pool_size=2, p=0.2):
    relud_batch_norm = slim.batch_norm(
        input, activation_fn=tf.nn.relu, scope=(scope + "/batchnorm"))
    conv = slim.conv2d(relud_batch_norm, num_features,
                       kernel_size, scope=(scope + "/conv"), activation_fn=None)
    dropout = slim.dropout(conv, p, scope=(scope + "/dropout"))
    max_pool = slim.max_pool2d(
        dropout, pool_size, stride=2, scope=(scope + "/maxpool"))
    return max_pool


def transition_up(input, num_features, scope, kernel_size=3, stride=2):
    return slim.conv2d_transpose(input, num_features, kernel_size,
                                 stride=stride, scope=scope, activation_fn=None)
