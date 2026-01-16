import os

from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import BORDER_LABEL, CONTENT_LABEL, BACKGROUND_LABEL, tf_count


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


def acc_per_label(y_true, y_pred, label):
    pred_mask = tf.argmax(y_pred, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    y_true = tf.cast(y_true, pred_mask.dtype)
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


def save_model_history_metrics(epochs, history):
    epochs = range(epochs)
    output_path = "./graphs"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_loss_history_metrics(epochs, history)
    save_acc_history_metrics(epochs, history)
    save_border_acc_history_metrics(epochs, history)
    save_content_acc_history_metrics(epochs, history)
    save_background_acc_history_metrics(epochs, history)
    save_acc_history_metrics_per_label(epochs, history)


def save_acc_history_metrics_per_label(epochs, history):
    val_border_metrics = history.history['val_border_acc']
    val_content_metrics = history.history['val_content_acc']
    val_background_metrics = history.history['val_background_acc']
    plt.figure()
    plt.plot(epochs, val_border_metrics, 'r', label='Validation border acc')
    plt.plot(epochs, val_content_metrics, 'g', label='Validation content acc')
    plt.plot(epochs, val_background_metrics, 'b', label='Validation background acc')
    plt.title('Training and Validation Border ACC')
    plt.xlabel('Epoch')
    plt.ylabel('Acc Value')
    plt.legend()
    plt.savefig('./graphs/acc_per_label.png')


def save_border_acc_history_metrics(epochs, history):
    metrics = history.history['border_acc']
    val_metrics = history.history['val_border_acc']
    plt.figure()
    plt.plot(epochs, metrics, 'r', label='Training border acc')
    plt.plot(epochs, val_metrics, 'b', label='Validation border acc')
    plt.title('Training and Validation Border ACC')
    plt.xlabel('Epoch')
    plt.ylabel('Acc Value')
    plt.legend()
    plt.savefig('./graphs/border_acc.png')


def save_content_acc_history_metrics(epochs, history):
    metrics = history.history['content_acc']
    val_metrics = history.history['val_content_acc']
    plt.figure()
    plt.plot(epochs, metrics, 'r', label='Training content acc')
    plt.plot(epochs, val_metrics, 'b', label='Validation content acc')
    plt.title('Training and Validation Content ACC')
    plt.xlabel('Epoch')
    plt.ylabel('Acc Value')
    plt.legend()
    plt.savefig('./graphs/content_acc.png')


def save_background_acc_history_metrics(epochs, history):
    metrics = history.history['background_acc']
    val_metrics = history.history['val_background_acc']
    plt.figure()
    plt.plot(epochs, metrics, 'r', label='Training background acc')
    plt.plot(epochs, val_metrics, 'b', label='Validation background acc')
    plt.title('Training and Validation Background ACC')
    plt.xlabel('Epoch')
    plt.ylabel('Acc Value')
    plt.legend()
    plt.savefig('./graphs/background_acc.png')


def save_acc_history_metrics(epochs, history):
    metrics = history.history['accuracy']
    val_metrics = history.history['val_accuracy']
    plt.figure()
    plt.plot(epochs, metrics, 'r', label='Training acc')
    plt.plot(epochs, val_metrics, 'b', label='Validation acc')
    plt.title('Training and Validation ACC')
    plt.xlabel('Epoch')
    plt.ylabel('Acc Value')
    plt.legend()
    plt.savefig('./graphs/acc.png')


def save_loss_history_metrics(epochs, history):
    metrics = history.history['loss']
    val_metrics = history.history['val_loss']
    plt.figure()
    plt.plot(epochs, metrics, 'r', label='Training loss')
    plt.plot(epochs, val_metrics, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig('./graphs/loss.png')
