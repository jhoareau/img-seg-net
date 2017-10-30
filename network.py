import tensorflow as tf
slim = tf.contrib.slim

from building_blocks import dense_block, transition_down, transition_up

def input_convolution(input, num_features, kernel_size=3):
    return slim.conv2d(input, num_features, kernel_size, scope='inputConv')

def output_convolution(input, num_features, kernel_size=3):
    conv = slim.conv2d(input, num_features, kernel_size, scope='outputConv')
    return slim.softmax(conv, scope='outputConv_softmax')

def net(input, PARAMS):
    net = input_convolution(input, PARAMS['num_features'])
    dense_1, _ = dense_block(net, PARAMS['dense_1']['num_layers'], PARAMS['num_features'], 'dense1')

    net = transition_down(dense_1, PARAMS['num_features'], 'td1')
    dense_2, _ = dense_block(net, PARAMS['dense_2']['num_layers'], PARAMS['num_features'], 'dense2')

    net = transition_down(dense_2, PARAMS['num_features'], 'td2')
    dense_3, _ = dense_block(net, PARAMS['dense_3']['num_layers'], PARAMS['num_features'], 'dense3')

    net = transition_down(dense_3, PARAMS['num_features'], 'td3')
    dense_4, _ = dense_block(net, PARAMS['dense_4']['num_layers'], PARAMS['num_features'], 'dense4')

    net = transition_down(dense_4, PARAMS['num_features'], 'td4')
    dense_5, _ = dense_block(net, PARAMS['dense_5']['num_layers'], PARAMS['num_features'], 'dense5')

    net = transition_down(dense_5, PARAMS['num_features'], 'td5')
    _, dense_deep_layers = dense_block(net, PARAMS['dense_deep']['num_layers'], PARAMS['num_features'], 'denseDeep', skipped=False)
    dense_output = tf.concat(axis=-1, values=dense_deep_layers, name='denseDeep/output')
    net = transition_up(dense_output, PARAMS['num_features'], 'tu1')

    net = tf.concat(axis=-1, values=[dense_5, net], name='skip1_up')
    net, _ = dense_block(net, PARAMS['dense_1_up']['num_layers'], PARAMS['num_features'], 'dense1up')
    net = transition_up(net, PARAMS['num_features'], 'tu2')

    net = tf.concat(axis=-1, values=[dense_4, net], name='skip2_up')
    net, _ = dense_block(net, PARAMS['dense_2_up']['num_layers'], PARAMS['num_features'], 'dense2up')
    net = transition_up(net, PARAMS['num_features'], 'tu3')

    net = tf.concat(axis=-1, values=[dense_3, net], name='skip3_up')
    net, _ = dense_block(net, PARAMS['dense_3_up']['num_layers'], PARAMS['num_features'], 'dense3up')
    net = transition_up(net, PARAMS['num_features'], 'tu4')

    net = tf.concat(axis=-1, values=[dense_2, net], name='skip4_up')
    net, _ = dense_block(net, PARAMS['dense_4_up']['num_layers'], PARAMS['num_features'], 'dense4up')
    net = transition_up(net, PARAMS['num_features'], 'tu5')

    net = tf.concat(axis=-1, values=[dense_1, net], name='skip5_up')
    net, _ = dense_block(net, PARAMS['dense_5_up']['num_layers'], PARAMS['num_features'], 'dense5up')

    net = output_convolution(net, PARAMS['num_features'])

    return net
