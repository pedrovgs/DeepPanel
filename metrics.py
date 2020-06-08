from keras import backend as K
import tensorflow as tf

from utils import BORDER_LABEL, CONTENT_LABEL, BACKGROUND_LABEL


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count


def acc_per_label(y_true, y_pred, label):
    pred_mask = tf.argmax(y_pred, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    y_true = K.cast(y_true, pred_mask.dtype)
    true_label_count = tf_count(y_true, label)
    properly_predicted_labels = tf.where(tf.equal(y_true, label), x=pred_mask, y=-1)
    properly_predicted_label_count = tf_count(properly_predicted_labels, label)
    return properly_predicted_label_count / true_label_count


def border_acc(y_true, y_pred):
    return acc_per_label(y_true, y_pred, BORDER_LABEL)


def content_acc(y_true, y_pred):
    return acc_per_label(y_true, y_pred, CONTENT_LABEL)


def background_acc(y_true, y_pred):
    return acc_per_label(y_true, y_pred, BACKGROUND_LABEL)
