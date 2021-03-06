import os
from jinja2 import Environment, FileSystemLoader
import tensorflow as tf
from tensorflow import keras
from utils import load_data_set, load_image_test, map_prediction_to_mask, compare_accuracy, labeled_prediction_to_image, \
    count_files_in_folder, files_in_folder

from metrics import iou_coef, dice_coef, border_acc, background_acc, content_acc


class PredictionResult:

    def __init__(self, page, pred, mask):
        self.page = page
        self.pred = pred
        self.mask = mask


def generate_output_template():
    output_images = files_in_folder("./output/")
    test_images = files_in_folder("./dataset/test/raw/")
    true_masks_images = files_in_folder("./dataset/test/segmentation_mask/")
    template_predictions = list()
    for i in range(len(output_images)):
        template_predictions.append(PredictionResult(test_images[i], output_images[i], true_masks_images[i]))
    loader = FileSystemLoader("./templates")
    env = Environment(loader=loader)
    template = env.get_template('index.html')
    template_output = template.render(predictions=template_predictions)
    reports_path = "./reports"
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)
    text_file = open(f"{reports_path}/index.html", "w")
    text_file.write(template_output)
    text_file.close()


if __name__ == "__main__":
    print(" - Loading saved model")
    tf.random.set_seed(11)
    custom_objects = {
        "border_acc": border_acc,
        "background_acc": background_acc,
        "content_acc": content_acc,
        "iou_coef": iou_coef,
        "dice_coef": dice_coef
    }
    model = keras.models.load_model("./model", custom_objects=custom_objects)

    print(" - Loading test data")
    testing_files_path = "./dataset/test/raw"
    testing_num_files = count_files_in_folder(testing_files_path)
    TESTING_BATCH_SIZE = testing_num_files
    raw_dataset = load_data_set()
    test_raw_dataset = raw_dataset['test']
    test = test_raw_dataset.map(load_image_test)
    test_dataset = test.batch(TESTING_BATCH_SIZE)
    for images, true_masks in test_dataset:
        pass  # Hack needed to be able to extrac images and true masks from map datasets
    images = images.numpy()
    true_masks = true_masks.numpy()
    print(f" - Test data loaded for {testing_num_files} images")

    print(" - Prediction started")
    predictions = model.predict(test_dataset)
    predicted_images_number = len(predictions)
    print(f" - Prediction finished for {predicted_images_number} images")

    print(f" - Let's transform predictions into labeled values.")
    labeled_predictions = []
    for image_index in range(predicted_images_number):
        prediction = predictions[image_index]
        predicted_mask = map_prediction_to_mask(prediction)
        labeled_predictions.append(predicted_mask)

    print(f" - Saving labeled images into ./output folder")
    predicted_index = 0
    output_path = "./output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for predicted_result in labeled_predictions:
        prediction_as_image = labeled_prediction_to_image(predicted_result)
        prediction_as_image.save(f"{output_path}{predicted_index:03d}.jpg")
        prediction_as_image.close()
        print(f"    - Image with index {predicted_index} saved.")
        predicted_index += 1
    print(f" - Generating sample output page")
    generate_output_template()
    print(f" - Images saved. Time to check accuracy metrics per label:")
    background_acc, border_acc, content_acc = compare_accuracy(true_masks, labeled_predictions)
    print(" - Accuracy measures per label:")
    print(f"    - Border label = {border_acc}")
    print(f"    - Content label = {content_acc}")
    print(f"    - Background label = {background_acc}")
    print(f" - Saving predicted masks as images")
