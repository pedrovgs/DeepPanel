import glob

from PIL import Image, ImageOps

from utils import IMAGE_SIZE

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
        size = IMAGE_SIZE, IMAGE_SIZE
        image = Image.open(filename)
        pixels = image.load()
        width, height = image.size
        if height >= width:
            delta_w = height - width
            delta_h = 0
        else:
            delta_w = 0
            delta_h = width - height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        image = ImageOps.expand(image, padding).resize(size, Image.NEAREST)
        pixels = image.load()
        width, height = image.size
        number_of_pixels_per_image = width * height
        background_counter = 0
        border_counter = 0
        content_counter = 0
        for x in range(width):
            for y in range(height):
                rgba_info = pixels[x, y]
                r_channel = rgba_info[0]
                g_channel = rgba_info[1]
                b_channel = rgba_info[2]
                if r_channel == 255:
                    border_counter += 1
                elif g_channel == 255:
                    content_counter += 1
                elif b_channel == 255:
                    background_counter += 1
                elif r_channel == 0 and g_channel == 0 and b_channel == 0:
                    background_counter += 1
                else:
                    print("ERROR: INVALID PIXEL")
        background_rate += background_counter / number_of_pixels_per_image
        border_rate += border_counter / number_of_pixels_per_image
        content_rate += content_counter / number_of_pixels_per_image
        image.close()
    print(f" - Percentage of background in files = {background_rate / number_of_images}")
    print(f" - Percentage of border in files = {border_rate / number_of_images}")
    print(f" - Percentage of content in files = {content_rate / number_of_images}")
