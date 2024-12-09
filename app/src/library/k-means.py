import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib;
import cv2
from PIL import Image
import os;
from math import sqrt
from imantics import Mask


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


# construct the argument parser and parse the arguments
# plt.matplotlib_fname()
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image_original = cv2.imread("../../samples/161511023716926557.jpg");

scale_percent = 10  # percent of original size
width = int(image_original.shape[1] * scale_percent / 100)
height = int(image_original.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
image = cv2.resize(image_original, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Image", image)

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixel_vals = image.reshape((-1, 3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# then perform k-means clustering wit h number of clusters defined as 3
# also random centres are initally chosed for k-means clustering
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
# print()
centers = np.uint8(centers)
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
binarized = 1.0 * (gray > 127)
print(binarized)
percent = []
for i in range(len(centers)):
    j = list(labels).count(i)
    j = j / (len(labels))
    percent.append(j)

plt.pie(percent, colors=np.array(centers / 255), labels=np.arange(len(centers)))
plt.show()

cv2.imshow("seg", binarized)

cv2.waitKey(0)
cv2.destroyAllWindows()
