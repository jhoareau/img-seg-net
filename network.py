import tensorflow as tf
slim = tf.contrib.slim

from building_blocks import dense_block, transition_down, transition_up, l2_reg, weight_initializer


def net(input, PARAMS, is_training=True):
    net = slim.conv2d(input, PARAMS['input_num_features'],
                      3, weights_initializer=weight_initializer,
                      weights_regularizer=l2_reg,
                      scope='inputConv', activation_fn=None)
    dense_down = list()
    for i in range(1, 6):
        dense_k, _ = dense_block(net, PARAMS['dense_{}'.format(
            i)]['num_layers'], PARAMS['num_features'], 'dense{}'.format(i), is_training=is_training)
        net = transition_down(
            dense_k, PARAMS['num_features'], 'td{}'.format(i), is_training=is_training)
        dense_down.append(dense_k)

    _, dense_deep_layers = dense_block(
        net, PARAMS['dense_bottleneck']['num_layers'], PARAMS['num_features'], 'denseBottleneck', skipped=False, is_training=is_training)
    net = tf.concat(axis=-1, values=dense_deep_layers,
                    name='denseBottleneck/output')

    for i in range(1, 6):
        net = transition_up(net, PARAMS['num_features'], 'tu{}'.format(i))
        net = tf.concat(
            axis=-1, values=[dense_down[-i], net], name=('skip{}_up'.format(i)))
        net, _ = dense_block(net, PARAMS['dense_{}_up'.format(
            i)]['num_layers'], PARAMS['num_features'], 'dense{}up'.format(i), is_training=is_training)

    return slim.conv2d(net, PARAMS['output_classes'], 1,
            weights_initializer=weight_initializer, weights_regularizer=l2_reg,
            scope='outputConv', activation_fn=None)
