import tensorflow as tf
slim = tf.contrib.slim

from building_blocks import dense_block, transition_down, transition_up

def input_convolution(input, num_features, kernel_size):
    return slim.conv2d(input, num_features, kernel_size, scope='inputConv')

def output_convolution(input, num_features, kernel_size):
    conv = slim.conv2d(input, num_features, kernel_size, scope='outputConv')
    return slim.softmax(conv, scope='outputConv_softmax')

def net(input, PARAMS):
    net = input_convolution(input, PARAMS['num_features'], PARAMS['kernel_size'])
    dense_1, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense1')
    # Skip connection 1
    concat_1 = tf.concat(axis=-1, values=[dense_1, net], name='skip1')

    net = transition_down(concat_1, PARAMS['num_features'], 'td1', PARAMS['kernel_size'])
    dense_2, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense2')
    concat_2 = tf.concat(axis=-1, values=[dense_2, net], name='skip2')

    net = transition_down(concat_2, PARAMS['num_features'], 'td2', PARAMS['kernel_size'])
    dense_3, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense3')
    concat_3 = tf.concat(axis=-1, values=[dense_3, net], name='skip3')

    net = transition_down(concat_3, PARAMS['num_features'], 'td3', PARAMS['kernel_size'])
    dense_4, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense4')
    concat_4 = tf.concat(axis=-1, values=[dense_4, net], name='skip4')

    net = transition_down(concat_4, PARAMS['num_features'], 'td4', PARAMS['kernel_size'])
    dense_5, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense5')
    concat_5 = tf.concat(axis=-1, values=[dense_5, net], name='skip5')

    net = transition_down(concat_5, PARAMS['num_features'], 'td5', PARAMS['kernel_size'])
    _, dense_deep_layers = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'denseDeep', skipped=False)
    dense_output = tf.concat(axis=-1, values=dense_deep_layers, name='denseDeep/output')
    net = transition_up(dense_output, PARAMS['num_features'], 'tu1', PARAMS['kernel_size'])

    net = tf.concat(axis=-1, values=[concat_5, net], name='skip1_up')
    net, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense1up')
    net = transition_up(net, PARAMS['num_features'], 'tu2', PARAMS['kernel_size'])

    net = tf.concat(axis=-1, values=[concat_4, net], name='skip2_up')
    net, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense2up')
    net = transition_up(net, PARAMS['num_features'], 'tu3', PARAMS['kernel_size'])

    net = tf.concat(axis=-1, values=[concat_3, net], name='skip3_up')
    net, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense3up')
    net = transition_up(net, PARAMS['num_features'], 'tu4', PARAMS['kernel_size'])

    net = tf.concat(axis=-1, values=[concat_2, net], name='skip4_up')
    net, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense4up')
    net = transition_up(net, PARAMS['num_features'], 'tu5', PARAMS['kernel_size'])

    net = tf.concat(axis=-1, values=[concat_1, net], name='skip5_up')
    net, _ = dense_block(net, PARAMS['num_features'], PARAMS['kernel_size'], 'dense5up')

    net = output_convolution(net, PARAMS['num_features'], PARAMS['kernel_size'])

    return net
