import os

from tensorflow import keras
from DeepPanel import files_in_folder
from utils import load_data_set, load_image_test, map_prediction_to_mask, compare_accuracy, labeled_prediction_to_image

from metrics import iou_coef, dice_coef, border_acc, background_acc, content_acc

if __name__ == "__main__":
    print(" - Loading saved model")
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
    testing_num_files = files_in_folder(testing_files_path)
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

    print(f" - Images saved. Time to check accuracy metrics per label:")
    background_acc, border_acc, content_acc = compare_accuracy(true_masks, labeled_predictions)
    print(" - Accuracy measures per label:")
    print(f"    - Border label = {border_acc}")
    print(f"    - Content label = {content_acc}")
    print(f"    - Background label = {background_acc}")
    print(f" - Saving predicted masks as images")
