import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
import os.path
import multiprocessing

IMAGE_SIZE = 224


def contains_invalid_values(x):
    return x != 0


def parse_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    mask_path = tf.strings.regex_replace(img_path, "raw", "segmentation_mask")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    BACKGROUND_LABEL = 0
    BORDER_LABEL = 1
    CONTENT_LABEL = 2
    # Transform mask colors into labels
    # We will assume whites 0 which should be assigned to the background label
    mask = tf.where(mask == 255, np.dtype('uint8').type(BACKGROUND_LABEL), mask)
    # Dark values will use label the background label
    mask = tf.where(mask == 29, np.dtype('uint8').type(BACKGROUND_LABEL), mask)
    # Intermediate values will act as the border
    mask = tf.where(mask == 76, np.dtype('uint8').type(BORDER_LABEL), mask)
    mask = tf.where(mask == 134, np.dtype('uint8').type(BORDER_LABEL), mask)
    # Brighter values will act as the content
    mask = tf.where(mask == 149, np.dtype('uint8').type(CONTENT_LABEL), mask)
    return {'image': image, 'segmentation_mask': mask}


def load_images_from_folder(folder):
    return tf.data.Dataset.list_files(folder + "raw/*.jpg").map(parse_image)


def load_data_set():
    return {
        'test': load_images_from_folder("./dataset/test/"),
        'train': load_images_from_folder("./dataset/training/")
    }


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize_with_pad(datapoint['image'], target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
    input_mask = tf.image.resize_with_pad(datapoint['segmentation_mask'], target_height=IMAGE_SIZE,
                                          target_width=IMAGE_SIZE)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize_with_pad(datapoint['image'], target_height=IMAGE_SIZE, target_width=IMAGE_SIZE)
    input_mask = tf.image.resize_with_pad(datapoint['segmentation_mask'], target_height=IMAGE_SIZE,
                                          target_width=IMAGE_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n    - Training finished for epoch {}\n'.format(epoch + 1))


def files_in_folder(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])


if __name__ == "__main__":
    print(" - Loading data")
    raw_dataset = load_data_set()
    train_raw_dataset = raw_dataset['train']
    test_raw_dataset = raw_dataset['test']
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    training_files_path = "./dataset/training/raw"
    training_num_files = files_in_folder(training_files_path)
    testing_files_path = "./dataset/test/raw"
    testing_num_files = files_in_folder(testing_files_path)
    TRAIN_LENGTH = training_num_files
    TESTING_LENGTH = testing_num_files
    EPOCHS = 20
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

    # Optionally show image and masks:
    for image, mask in train.take(1):
        sample_image, sample_mask = image, mask
    # display([sample_image, sample_mask])

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
                  metrics=['accuracy'])
    tf.keras.utils.plot_model(model, show_shapes=True)

    print(" - Starting training stage")
    model_history = model.fit(train_dataset,
                              epochs=EPOCHS,
                              validation_data=test_dataset,
                              use_multiprocessing=True,
                              workers=CORES_COUNT,
                              callbacks=[DisplayCallback()])
    print(" - Training finished")
    show_predictions(test_dataset, 5)
