import tensorflow as tf
import matplotlib.pyplot as plt
import io as imp

# Remap colours
_cmap = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (0, 0, 192), (128, 128, 0),
         (192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0), (0, 128, 192), (0, 0, 0)]
_mask_labels = {0: 'sky', 1: 'building', 2: 'column_pole', 3: 'road',
                4: 'sidewalk', 5: 'tree', 6: 'sign', 7: 'fence', 8: 'car',
                9: 'pedestrian', 10: 'byciclist', 11: 'void'}


def colorize(grayimg):
    # Cast to integer to allow to function as index
    gray_int = tf.cast(grayimg, tf.int32)
    gray_remapped = tf.gather(_cmap, gray_int)  # Map gray to rgb
    gray_sq = tf.squeeze(gray_remapped)  # Remove the dimension of 1
    return tf.cast(gray_sq, tf.float32)  # Return as a float

# Legend


def legend():
    blocksize = 0.1
    plt.figure(figsize=(5, 3))
    for i in range(len(_cmap)):
        xoffset = 0
        yoffset = 0
        if(i > 5):
            xoffset = 0.5
            yoffset = 6 * blocksize
        clr = (_cmap[i][0] / 255, _cmap[i][1] / 255, _cmap[i][2] / 255)
        square = plt.Rectangle(
            (0 + xoffset, i * 0.1 - yoffset), width=blocksize, height=blocksize, fc=clr)
        plt.text(0.2 + xoffset, 0.03 + i * 0.1 -
                 yoffset, _mask_labels[i], fontsize=10)
        plt.gca().add_patch(square)
    plt.ylim([0, 0.6])
    plt.axis('off')
    buf = imp.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


buffer = legend()
# To TF image
legend = tf.image.decode_png(buffer.getvalue(), channels=4)
legend = tf.expand_dims(legend, 0)
