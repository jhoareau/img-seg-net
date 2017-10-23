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

def transition_down_and_block(input, dims, scope_postfix):
    transition_layer = transition_down(input, 'td_%s' % scope_postfix)
    dense_block_output = dense_block(transition_layer, [1, 1], 'dense_down_%s' % scope_postfix)

    return dense_block_output

def block_and_transition_up(input, dims, scope_postfix):
    dense_block_output = dense_block(input, [1, 1], 'dense_up_%s' % scope_postfix)
    transition_layer = transition_up(dense_block_output, 'tu_%s' % scope_postfix)

    return transition_layer

def net():
    net =
    net = input_convolution(input)
    net = transition_down(input)
    return net
