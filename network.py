import tensorflow.contrib.slim as slim
from layers import dense_block, transition_down, transition_up

def input_convolution(input, num_features, kernel_size):
    return slim.conv2d(input, num_features, kernel_size, scope='inputConv')

def output_convolution(input, num_features, kernel_size):
    conv = slim.conv2d(input, num_features, kernel_size, scope='outputConv')
    return slim.softmax(conv, scope='outputConv_softmax')

def transition_down_and_block(input, num_layers, num_features, kernel_size, pool_size, scope_postfix):
    transition_layer = transition_down(input, 'td_%s' % scope_postfix)
    x, dense_block_output = dense_block(transition_layer, num_layers, num_features, 'denseDown_%s' % scope_postfix)

    return dense_block_output

def block_and_transition_up(input, num_layers, num_features, kernel_size, pool_size, scope_postfix):
    x, dense_block_output = dense_block(input, num_layers, num_features, 'dense_up_%s' % scope_postfix)
    transition_layer = transition_up(dense_block_output, 'tu_%s' % scope_postfix)

    return transition_layer

def net(input):
    net = input_convolution(input)
    dense_1 = dense_block(net)
    # Skip connection 1
    concat_1 = tf.concat(axis=-1, values=[dense_1, net])

    net = transition_down(concat_1)
    dense_2 = dense_block(net)
    concat_2 = tf.concat(axis=-1, values=[dense_2, net])

    net = transition_down(concat_2)
    net = dense_block(net)
    net = transition_up(net)

    net = tf.concat(axis=-1, values=[concat_2, net])
    net = dense_block(net)
    net = transition_up(net)

    net = tf.concat(axis=-1, values=[concat_1, net])
    net = dense_block(net)

    net = output_convolution(net)

    return net
