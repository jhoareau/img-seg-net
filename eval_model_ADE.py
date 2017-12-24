import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from network import net
import json
from data_utils_ADE import *
from colorise_camvid import colorize, legend, _mask_labels
slim = tf.contrib.slim
from tensorflow.python.ops import math_ops, array_ops
def intersection_over_union(ground_truths, predictions, num_classes):
    with tf.variable_scope('iou'):
        batch_size = ground_truths.shape[0]
        iou_array = list()
        for i in range(batch_size):
            gt = tf.reshape(ground_truths[i], [-1])
            pr = tf.reshape(predictions[i], [-1])
            confusion_matrix = tf.confusion_matrix(gt, pr, num_classes, name='cm/' + str(i))
            sum_over_row = math_ops.to_float(math_ops.reduce_sum(confusion_matrix, 0))
            sum_over_col = math_ops.to_float(math_ops.reduce_sum(confusion_matrix, 1))
            cm_diag = math_ops.to_float(array_ops.diag_part(confusion_matrix))
            denominator = sum_over_row + sum_over_col - cm_diag

            # The mean is only computed over classes that appear in the
            # label or prediction tensor. If the denominator is 0, we need to
            # ignore the class.
            num_valid_entries = math_ops.reduce_sum(math_ops.cast(
              math_ops.not_equal(denominator, 0), dtype=tf.float32))

            # If the value of the denominator is 0, set it to 1 to avoid
            # zero division.
            denominator = array_ops.where(
              math_ops.greater(denominator, 0),
              denominator,
              array_ops.ones_like(denominator))
            iou = math_ops.div(cm_diag, denominator)
            iou_array.append(iou)

        iou_array = tf.stack(iou_array)
        return tf.reduce_mean(iou_array, axis=0), tf.reduce_mean(iou_array) 



n_images = 10 # total 532
classes=2
batch_size, height, width, nchannels = n_images, 360, 480, 3
final_resized = 224
model_version = 56
datasetfilename="validationset_ADE.tfrec"
_mask_labels_ADE = {0: 'nonhuman', 1: 'human'}


with open('model_parameters.json') as params:
    params_dict = json.load(params)[repr(model_version)]
params_dict['input_num_features'] = 48
params_dict['output_classes'] = classes

if not os.path.isfile(datasetfilename):
    tfrec_dump("validation", datasetfilename)
tfsdataset = slim_dataset(datasetfilename, n_images) 

gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# Training loop
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
    log_dir = 'val_ade'
    # We load a batch and reshape to tensor
    xbatch, ybatch = batch(
        tfsdataset, batch_size=batch_size, height=height, width=width, resized=final_resized)
    input_batch = tf.reshape(xbatch, shape=(batch_size, final_resized, final_resized, 3))
    ground_truth_batch = tf.reshape(ybatch, shape=(batch_size, final_resized, final_resized, 1))

    # Obtain the prediction
    predictions = net(input_batch, params_dict, is_training=False)
   
    # We calculate the loss
    one_hot_labels = slim.one_hot_encoding(
        tf.squeeze(ground_truth_batch),
        params_dict['output_classes'])
    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels)

    predim = tf.nn.softmax(predictions)
    predimmax = tf.expand_dims(tf.cast(tf.argmax(predim, axis=3), tf.float32), -1)
        
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)
    
    yb = tf.cast(tf.divide(ground_truth_batch, classes), tf.float32)
    
    predim = tf.nn.softmax(predictions)
    predimmax = tf.expand_dims(
        tf.cast(tf.argmax(predim, axis=3), tf.float32), -1)
    predimmaxdiv = tf.divide(tf.cast(predimmax, tf.float32), classes)
    
    ediff = tf.abs(tf.subtract(yb, predimmaxdiv))
    norm_ediff = tf.ceil(ediff)
    accuracy = tf.reduce_mean(tf.cast(norm_ediff, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    annots=tf.concat([tf.multiply(input_batch, yb),tf.multiply(input_batch, predimmaxdiv)],2)
    img_and_err=tf.concat([input_batch,tf.image.grayscale_to_rgb(norm_ediff)],2)
    aio=tf.concat([img_and_err,annots],1)
    tf.summary.image("All_in_one", aio, max_outputs=n_images)
    
    
    iou_array, mean_iou = intersection_over_union(ground_truth_batch, predimmax, params_dict['output_classes'])
    tf.summary.scalar('mean_IoU', mean_iou)
    class_labels = tf.convert_to_tensor(np.array(list(_mask_labels_ADE.values())), tf.string)
    iou_per_class = tf.stack([class_labels, tf.as_string(iou_array, precision=2)], axis=1)
    tf.summary.text('IoU per class', iou_per_class)

    slim.evaluation.evaluate_once(
        '',
        'train_ade/model.ckpt-46176',
        'val_ade'
    )
