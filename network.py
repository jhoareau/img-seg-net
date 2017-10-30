import tensorflow as tf
slim = tf.contrib.slim

from building_blocks import dense_block, transition_down, transition_up

def input_convolution(input, num_features, kernel_size):
    return slim.conv2d(input, num_features, kernel_size, scope='inputConv')

def output_convolution(input, num_features, kernel_size):
    conv = slim.conv2d(input, num_features, kernel_size, scope='outputConv')
    return slim.softmax(conv, scope='outputConv_softmax')

def net(input, PARAMS):
    net = input_convolution(input, PARAMS['input_convolution']['num_features'], PARAMS['input_convolution']['kernel_size'])
    dense_1, _ = dense_block(net, PARAMS['dense_1']['num_features'], PARAMS['dense_1']['kernel_size'], 'dense1')
    # Skip connection 1
    concat_1 = tf.concat(axis=-1, values=[dense_1, net], name='skip1')

    net = transition_down(concat_1, PARAMS['transition_down_1']['num_features'], 'td1', PARAMS['transition_down_1']['kernel_size'])
    dense_2, _ = dense_block(net, PARAMS['dense_2']['num_features'], PARAMS['dense_2']['kernel_size'], 'dense2')
    concat_2 = tf.concat(axis=-1, values=[dense_2, net], name='skip2')

    net = transition_down(concat_2, PARAMS['transition_down_2']['num_features'], 'td2', PARAMS['transition_down_2']['kernel_size'])
    _, layers = dense_block(net, PARAMS['dense_deep']['num_features'], PARAMS['dense_deep']['kernel_size'], 'denseDeep')
    dense_output = tf.concat(axis=-1, values=layers, name='dense_deep')
    net = transition_up(dense_output, PARAMS['transition_up_1']['num_features'], 'tu1', PARAMS['transition_up_1']['kernel_size'])

    net = tf.concat(axis=-1, values=[concat_2, net])
    net, _ = dense_block(net, PARAMS['dense_1_up']['num_features'], PARAMS['dense_1_up']['kernel_size'], 'dense1up')
    net = transition_up(net, PARAMS['transition_up_2']['num_features'], 'tu2', PARAMS['transition_up_2']['kernel_size'])

    net = tf.concat(axis=-1, values=[concat_1, net])
    net, _ = dense_block(net, PARAMS['dense_2_up']['num_features'], PARAMS['dense_2_up']['kernel_size'], 'dense2up')

    net = output_convolution(net, PARAMS['output_convolution']['num_features'], PARAMS['output_convolution']['kernel_size'])

    return net
