import tensorflow.contrib.slim as slim
import tensorflow as tf

l2_reg = slim.l2_regularizer(0.0001)
regularizers = {"beta" : slim.l2_regularizer(0.0001), "gamma": slim.l2_regularizer(0.0001)}
# weight_initializer = tf.contrib.layers.xavier_initializer()
weight_initializer = tf.contrib.keras.initializers.he_uniform()

def batch_wise_batch_norm(x, scope):
    with tf.name_scope(scope):
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        x = tf.subtract(x, batch_mean)
        x = tf.div(x, tf.sqrt(batch_var) + 1e-6)
        x = tf.nn.relu(x, "BatchNormRelu")
        return x

def skipped_dense_block(x, num_layers, num_features, scope, p=0.2, is_training=True):
    layers = []
    for i in range(num_layers):
        layer = create_layer(x, num_features, p=p,
                             scope=(scope + "/layer" + str(i)), is_training=is_training)
        layers.append(layer)
        x = tf.concat(axis=-1, values=[x, layer],
                      name=(scope + "/skip" + str(i)))
    return x

def nonskipped_dense_block(x, num_layers, num_features, scope, p=0.2, is_training=True):
    layers = []
    for i in range(num_layers):
        layer = create_layer(x, num_features, p=p,
                             scope=(scope + "/layer" + str(i)), is_training=is_training)
        layers.append(layer)
        if (i == num_layers - 1):
            continue
        x = tf.concat(axis=-1, values=[x, layer],
                      name=(scope + "/skip" + str(i)))
    return tf.concat(axis=-1, values=layers,
                    name=(scope + '/output'))

def create_layer(input, num_features, scope, kernel_size=3, p=0.2, is_training=True):
    relud_batch_norm = batch_wise_batch_norm(input, scope + "/batchnorm")
    # relud_batch_norm = slim.batch_norm(input, activation_fn=tf.nn.relu, 
    # param_regularizers=regularizers, scope=(scope + "/batchnorm"))
    conv = slim.conv2d(relud_batch_norm, num_features,
                       kernel_size, weights_initializer=weight_initializer,
                       weights_regularizer=l2_reg,
                       scope=(scope + "/conv"), activation_fn=None)
    dropout = slim.dropout(conv, keep_prob=1-p, scope=(scope + "/dropout"), is_training=is_training)
    return dropout


def transition_down(input, scope, kernel_size=1, pool_size=2, p=0.2, is_training=True):
    relud_batch_norm = batch_wise_batch_norm(input, scope + "/batchnorm")
    # relud_batch_norm = slim.batch_norm(input, activation_fn=tf.nn.relu,
    #      param_regularizers=regularizers, scope=(scope + "/batchnorm"))
    conv = slim.conv2d(relud_batch_norm, input.shape[-1],
                       kernel_size, weights_initializer=weight_initializer,
                       weights_regularizer=l2_reg,
                       scope=(scope + "/conv"), activation_fn=None)
    dropout = slim.dropout(conv, keep_prob=1-p, scope=(scope + "/dropout"), is_training=is_training)
    max_pool = slim.max_pool2d(
        dropout, pool_size, stride=2, scope=(scope + "/maxpool"))
    return max_pool


def transition_up(input, scope, kernel_size=3, stride=2):
    return slim.conv2d_transpose(input, input.shape[-1], kernel_size,
                                 weights_initializer=weight_initializer,
                                 weights_regularizer=l2_reg,
                                 stride=stride, scope=scope, activation_fn=None)
