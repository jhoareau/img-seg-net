import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
def intersection_over_union(ground_truths, predictions, num_classes, weights):
    batch_size = ground_truths.shape[0]
    iou_array = list()
    for i in range(batch_size):
        gt = tf.reshape(ground_truths[i], [-1])
        pr = tf.reshape(predictions[i], [-1])
        we = tf.reshape(weights[i], [-1])
        confusion_matrix = tf.confusion_matrix(gt, pr, num_classes, weights=we)
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
