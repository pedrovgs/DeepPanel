import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output

IMAGE_SIZE = 224
BACKGROUND_LABEL = 0
BORDER_LABEL = 1
CONTENT_LABEL = 2


def display(display_list):
    clear_output(wait=True)
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions_compared_to_real_data(images, true_masks, predictions):
    for image_index in range(len(predictions)):
        image = images[image_index]
        true_mask = true_masks[image_index]
        labeled_prediction = predictions[image_index]
        display([image, true_mask, labeled_prediction])


def parse_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    mask_path = tf.strings.regex_replace(img_path, "raw", "segmentation_mask")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
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


def load_images_from_folder(folder, shuffle=True):
    files = tf.data.Dataset.list_files(folder + "raw/*.jpg", shuffle=shuffle)
    return files.map(parse_image)


def load_data_set():
    return {
        # Even when shuffle is recommended we don't want to shuffle the test dataset in order to be
        # able to easily interpret the prediction result using the order as the index and assign the
        # prediction index to the test image position inside the test folder.
        'test': load_images_from_folder("./dataset/test/", shuffle=False),
        'train': load_images_from_folder("./dataset/training/")
    }


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


def predicted_pixel_to_class(x):
    return np.argmax(x)


# TODO: This makes us go really slow when checking accuracy.
# Review how we can make this faster in the future using arrays.
def map_prediction_to_mask(predicted_image):
    predicted_mask = list()
    for x in predicted_image:
        predicted_mask_per_x = []
        for y in x:
            predicted_mask_per_x.append(predicted_pixel_to_class(y))
        predicted_mask.append(predicted_mask_per_x)
    return np.array(predicted_mask)


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


def files_in_folder(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])


# TODO: This is really slow. Should we avoid index access and use iterators instead?
def compare_accuracy_per_label(true_mask, predicted_mask):
    total_background_labels = 0
    total_border_labels = 0
    total_content_labels = 0
    properly_predicted_background_pixels = 0
    properly_predicted_border_pixels = 0
    properly_predicted_content_pixels = 0
    for x in range(IMAGE_SIZE):
        for y in range(IMAGE_SIZE):
            prediction_label_per_pixel = int(predicted_mask[x][y])
            mask_label_per_pixel = int(true_mask[x][y])
            correct_prediction = prediction_label_per_pixel == mask_label_per_pixel
            if mask_label_per_pixel == BACKGROUND_LABEL:
                total_background_labels += 1
                if correct_prediction:
                    properly_predicted_background_pixels += 1
            if mask_label_per_pixel == BORDER_LABEL:
                total_border_labels += 1
                if correct_prediction:
                    properly_predicted_border_pixels += 1
            if mask_label_per_pixel == CONTENT_LABEL:
                total_content_labels += 1
                if correct_prediction:
                    properly_predicted_content_pixels += 1

    background_accuracy = properly_predicted_background_pixels / total_background_labels
    border_accuracy = properly_predicted_border_pixels / total_border_labels
    content_accuracy = properly_predicted_content_pixels / total_content_labels
    return background_accuracy, border_accuracy, content_accuracy


def compare_accuracy(true_masks, predictions):
    background_acc = 0.0
    border_acc = 0.0
    content_acc = 0.0
    for index in range(len(predictions)):
        print(f"   - Checking accuracy for image with index {index}")
        partial_back_acc, partial_border_acc, partial_content_acc = compare_accuracy_per_label(true_masks[index],
                                                                                               predictions[index])
        background_acc += partial_back_acc
        border_acc += partial_border_acc
        content_acc += partial_content_acc
        index += 1
    pred_num = len(predictions)
    return background_acc / pred_num, border_acc / pred_num, content_acc / pred_num


def label_to_rgb(labeled_pixel):
    if labeled_pixel == BACKGROUND_LABEL:
        return 0
    if labeled_pixel == CONTENT_LABEL:
        return 127
    if labeled_pixel == BORDER_LABEL:
        return 255


def labeled_prediction_to_image(predicted_result):
    color_matrix = np.vectorize(label_to_rgb)(predicted_result)
    return Image.fromarray(np.uint8(color_matrix))
