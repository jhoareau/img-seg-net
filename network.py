import tensorflow as tf
slim = tf.contrib.slim

from building_blocks import *

def net(input, PARAMS, is_training=True):
    net = slim.conv2d(input, PARAMS['input_num_features'],
                      3, weights_initializer=weight_initializer,
                      weights_regularizer=l2_reg,
                      scope='inputConv', activation_fn=None)
    dense_down = list()
    for i in range(1, 6):
        dense_k = skipped_dense_block(net, PARAMS['dense_{}'.format(i)]['num_layers'],
            PARAMS['num_features'], 'dense{}'.format(i), is_training=is_training)
        net = transition_down(dense_k, 'td{}'.format(i), is_training=is_training)
        dense_down.append(dense_k)

    net = nonskipped_dense_block(
        net, PARAMS['dense_bottleneck']['num_layers'], PARAMS['num_features'], 'denseBottleneck', is_training=is_training)

    for i in range(1, 6):
        net = transition_up(net, 'tu{}'.format(i))
        net = tf.concat(
            axis=-1, values=[dense_down[-i], net], name=('skip{}_up'.format(i)))
        if (i < 5):
            net = nonskipped_dense_block(net, PARAMS['dense_{}_up'.format(i)]['num_layers'],
                PARAMS['num_features'], 'dense{}up'.format(i), is_training=is_training)
        else:
            # Last upsampling dense block has a skip connection
            net = skipped_dense_block(net, PARAMS['dense_{}_up'.format(i)]['num_layers'],
                PARAMS['num_features'], 'dense{}up'.format(i), is_training=is_training)

    return slim.conv2d(net, PARAMS['output_classes'], 1,
            weights_initializer=weight_initializer, weights_regularizer=l2_reg,
            scope='outputConv', activation_fn=None)
