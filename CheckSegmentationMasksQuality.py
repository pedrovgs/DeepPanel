from PIL import Image
import glob
import sys


def adjust_pixel_values(pixels, x, y):
    rgba_info = pixels[x, y]
    r_channel = rgba_info[0]
    g_channel = rgba_info[1]
    b_channel = rgba_info[2]
    if r_channel > g_channel and r_channel > b_channel:
        pixels[x, y] = (255, 0, 0)
    if g_channel > r_channel and g_channel > b_channel:
        pixels[x, y] = (0, 255, 0)
    if b_channel > g_channel and b_channel > r_channel:
        pixels[x, y] = (0, 0, 255)


def contains_invalid_values(image_info):
    image_file = image_info['image']
    pixels = image_file.load()
    width, height = image_file.size
    weird_colors = False
    for x in range(width):
        for y in range(height):
            rgba_info = pixels[x, y]
            r_channel = rgba_info[0]
            g_channel = rgba_info[1]
            b_channel = rgba_info[2]
            if r_channel != 0 and r_channel != 255:
                adjust_pixel_values(pixels, x, y)
                weird_colors = True
            if g_channel != 0 and g_channel != 255:
                adjust_pixel_values(pixels, x, y)
                weird_colors = True
            if b_channel != 0 and b_channel != 255:
                adjust_pixel_values(pixels, x, y)
                weird_colors = True
    if weird_colors:
        print("Fixing colors...")
        image_file.save(image_info['filename'])
    return weird_colors


if __name__ == "__main__":
    image_list = []
    folder = sys.argv[1] if len(sys.argv) > 1 else "./dataset/training/segmentation_mask/"
    for filename in glob.glob(f'{folder}*.png'):
        im = Image.open(filename)
        image_list.append({'image': im, 'filename': filename})
    for image in image_list:
        print(f"Checking image {image['filename']}")
        if contains_invalid_values(image):
            print(f"The image {image['filename']} contains invalid rgb values")
