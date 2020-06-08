import glob

from PIL import Image

from utils import BACKGROUND_LABEL, BORDER_LABEL, CONTENT_LABEL

if __name__ == "__main__":
    background_rate = 0
    border_rate = 0
    content_rate = 0
    folder = "./dataset/training/segmentation_mask/"
    images = glob.glob(f'{folder}*.png')
    number_of_images = len(images)
    print(f"Starting training data analysis for {number_of_images} images")
    for filename in images:
        print(f"Analysing image {filename}")
        ## TODO RESIZE
        image = Image.open(filename)
        pixels = image.load()
        width, height = image.size
        number_of_pixels_per_image = width * height
        background_counter = 0
        border_counter = 0
        content_counter = 0
        for x in range(width):
            for y in range(height):
                rgba_info = pixels[x, y]
                r_channel = rgba_info[BACKGROUND_LABEL]
                g_channel = rgba_info[BORDER_LABEL]
                b_channel = rgba_info[CONTENT_LABEL]
                if r_channel == 255:
                    background_counter += 1
                elif g_channel == 255:
                    border_counter += 1
                elif b_channel == 255:
                    content_counter += 1
                else:
                    print("ERROR: INVALID PIXEL")
        background_rate += background_counter / number_of_pixels_per_image
        border_rate += border_counter / number_of_pixels_per_image
        content_rate += content_counter / number_of_pixels_per_image
        image.close()
    print(f" - Percentage of background in files = {background_rate / number_of_images}")
    print(f" - Percentage of border in files = {border_rate / number_of_images}")
    print(f" - Percentage of content in files = {content_rate / number_of_images}")
