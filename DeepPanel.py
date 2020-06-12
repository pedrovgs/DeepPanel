import os
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import multiprocessing

from metrics import iou_coef, dice_coef, border_acc, content_acc, background_acc
from utils import load_data_set, load_image_train, load_image_test, files_in_folder, IMAGE_SIZE


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n    - Training finished for epoch {}\n'.format(epoch + 1))


if __name__ == "__main__":
    print(" - Loading data")
    tf.random.set_seed(11)
    raw_dataset = load_data_set()
    train_raw_dataset = raw_dataset['train']
    test_raw_dataset = raw_dataset['test']
    training_files_path = "./dataset/training/raw"
    training_num_files = files_in_folder(training_files_path)
    testing_files_path = "./dataset/test/raw"
    testing_num_files = files_in_folder(testing_files_path)
    TRAIN_LENGTH = training_num_files
    TESTING_LENGTH = testing_num_files
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    EPOCHS = 15
    BUFFER_SIZE = TRAIN_LENGTH
    TRAINING_BATCH_SIZE = 20
    TESTING_BATCH_SIZE = TESTING_LENGTH
    CORES_COUNT = multiprocessing.cpu_count()
    print(" - Transforming data into vectors the model can understand")
    print(f"   - Training dataset size = {TRAIN_LENGTH}")
    print(f"   - Testing dataset size = {TESTING_LENGTH}")
    print(f"   - Training batch size = {TRAINING_BATCH_SIZE}")
    print(f"   - Testing batch size = {TESTING_BATCH_SIZE}")
    print(f"   - Epochs = {EPOCHS}")
    print(f"   - Buffer size = {BUFFER_SIZE}")
    print(f"   - Core's count = {CORES_COUNT}")

    train = train_raw_dataset.map(load_image_train, AUTOTUNE)
    test = test_raw_dataset.map(load_image_test)
    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(TRAINING_BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test.batch(TESTING_BATCH_SIZE)

    print(" - Creating the model")
    OUTPUT_CHANNELS = 3
    base_model = tf.keras.applications.MobileNetV2(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], include_top=False)
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 112x112
        'block_3_expand_relu',  # 56x56
        'block_6_expand_relu',  # 28x28
        'block_13_expand_relu',  # 14x14
        'block_16_project',  # 7x7
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False
    up_stack = [
        pix2pix.upsample(576, 3),  # 7x7 -> 14x14
        pix2pix.upsample(192, 3),  # 14x14 -> 28x28
        pix2pix.upsample(144, 3),  # 28x28 -> 56x56
        pix2pix.upsample(96, 3),  # 56x56 -> 112x112
    ]


    def unet_model(output_channels):
        inputs = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
        x = inputs

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same')  # 64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


    model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[
                      'accuracy',
                      border_acc,
                      content_acc,
                      background_acc,
                      iou_coef,
                      dice_coef])
    tf.keras.utils.plot_model(model, show_shapes=True)

    print(" - Starting training stage")
    model_history = model.fit(train_dataset,
                              epochs=EPOCHS,
                              validation_data=test_dataset,
                              use_multiprocessing=True,
                              workers=CORES_COUNT,
                              callbacks=[DisplayCallback()])
    print(" - Training finished, saving model into ./model")
    output_path = "./model"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.save(output_path)
    print(" - Model updated and saved")
