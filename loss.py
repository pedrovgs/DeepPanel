import tensorflow as tf

weights = tf.constant([1, 2, 1])


def weight_sparse_categorical_crossentropy(y_true, y_pred):
    # get the prediction from the final softmax layer:
    pred_idx = tf.argmax(y_pred, axis=1, output_type=tf.int32)

    # stack these so we have a tensor of [[predicted_i, actual_i], ...,] for each i in batch
    indices = tf.stack([tf.reshape(pred_idx, (-1,)),
                        tf.reshape(tf.cast(y_true, tf.int32), (-1,))
                        ], axis=1)

    # use tf.gather_nd() to convert indices to the appropriate weight from our matrix [w_i, ...] for each i in batch
    batch_weights = tf.gather_nd(weights, indices)

    return batch_weights * tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
