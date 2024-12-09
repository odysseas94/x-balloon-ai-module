import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib;
import cv2
from PIL import Image
from math import sqrt
from imantics import Mask


class KMeansCluster:
    actual_image = None
    output_binary_image = None
    saved_path: str = ""

    def __init__(self, actual_image, saved_path):
        self.actual_image = actual_image
        self.saved_path = saved_path
        self.output_binary_image = None

    def detect(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        k = 3
        image = cv2.cvtColor(self.actual_image, cv2.COLOR_BGR2RGB)
        pixel_vals = image.reshape((-1, 3))

        # Convert to float type
        pixel_vals = np.float32(pixel_vals)
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        wanted_channel = closest_color(centers, [233, 150, 122])
        wanted_channels = []

        for rgb in centers:
            if numpy.array_equal(rgb, wanted_channel):
                wanted_channels.append([0, 0, 0])
            else:
                wanted_channels.append([255, 255, 255])

        former_centers = np.uint8(numpy.array(wanted_channels))

        segmented_data = former_centers[labels.flatten()]

        # reshape data into the original image dimensions
        segmented_image = segmented_data.reshape((image.shape))
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        self.output_binary_image = 1.0 * (gray > 127)
        self.output_binary_image = np.logical_not(self.output_binary_image)
        self.output_binary_image = self.output_binary_image * 255
        # print("output_binary_image",self.output_binary_image.shape)

        cv2.imwrite(self.saved_path, self.output_binary_image)


def remove_channels(data, rgb: list, defualt_replace_color=[255, 255, 255]):
    r1, g1, b1 = rgb
    r2, g2, b2 = defualt_replace_color  # Value that we want to replace it with
    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]
    return data


def closest_color(all_colors, rgb):
    r, g, b = rgb
    color_diffs = []
    for color in all_colors:
        cr, cg, cb = color
        color_diff = sqrt(abs(r - cr) ** 2 + abs(g - cg) ** 2 + abs(b - cb) ** 2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]
