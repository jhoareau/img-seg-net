import tensorflow.contrib.slim as slim

# from layers import dense_block, transition_down, transition_up
def dense_block(input, dims, scope):
    return slim.conv2d(input, 1, [3, 3], scope=scope)

def transition_down(input, scope):
    return slim.conv2d(input, 1, [3, 3], scope=scope)

def transition_up(input, scope):
    return slim.conv2d_transpose(input, 1, [3, 3], scope=scope)

def input_convolution(input):
    return slim.conv2d(input, 1, [3, 3], scope='input_conv')

def output_convolution(input):
    conv = slim.conv2d(input, 1, [3, 3], scope='output_conv')
    return tf.nn.softmax(conv)

def transition_down_and_block(input, dims, scope_postfix):
    transition_layer = transition_down(input, 'td_%s' % scope_postfix)
    dense_block_output = dense_block(transition_layer, [1, 1], 'dense_down_%s' % scope_postfix)

    return dense_block_output

def block_and_transition_up(input, dims, scope_postfix):
    dense_block_output = dense_block(input, [1, 1], 'dense_up_%s' % scope_postfix)
    transition_layer = transition_up(dense_block_output, 'tu_%s' % scope_postfix)

    return transition_layer

def net(input):
    net = input_convolution(input)
    dense_1 = dense_block(net)
    # Skip connection 1
    concat_1 = [dense_1 net]

    net = transition_down(concat_1)
    dense_2 = dense_block(net)
    concat_2 = [dense_2 net]

    net = transition_down(concat_2)
    net = dense_block(net)
    net = transition_up(net)

    net = [concat_2 net]
    net = dense_block(net)
    net = transition_up(net)

    net = [concat_1 net]
    net = dense_block(net)

    net = output_convolution(net)

    return net
