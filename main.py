import tensorflow as tf
from network import net

params_dict = {
    'input_convolution': {
        'kernel_size': 3,
        'num_features': 3
    },
    'output_convolution': {
        'kernel_size': 3,
        'num_features': 3
    },
    'transition_down_1': {
        'kernel_size': 3,
        'num_features': 3
    },
    'transition_down_2': {
        'kernel_size': 3,
        'num_features': 3
    },
    'transition_up_1': {
        'kernel_size': 3,
        'num_features': 3
    },
    'transition_up_2': {
        'kernel_size': 3,
        'num_features': 3
    },
    'dense_1': {
        'kernel_size': 3,
        'num_features': 3
    },
    'dense_1': {
        'kernel_size': 3,
        'num_features': 3
    },
    'dense_2': {
        'kernel_size': 3,
        'num_features': 3
    },
    'dense_deep': {
        'kernel_size': 3,
        'num_features': 3
    },
    'dense_1_up': {
        'kernel_size': 3,
        'num_features': 3
    },
    'dense_2_up': {
        'kernel_size': 3,
        'num_features': 3
    }
}

with tf.Session() as sess:
    net([1], params_dict)
    tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    train_writer = tf.summary.FileWriter('train', sess.graph)
    tf.global_variables_initializer().run()
