import os

import cv2, numpy as np
from sklearn.cluster import KMeans


def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect


dir = "E:\\Users\\Projects\\x-ballon\\HERPATOLOGY\\annotation_2nd_round"
# Load image and convert to a list of pixels
images = []
for file in os.listdir(dir):
    if file.endswith(".jpg") or file.endswith(".png"):
        print("found")

        images.append(cv2.imread(os.path.join(dir, file)))
total_avg = [0, 0, 0];
all = [];
length = len(images)

for image in images:
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    total_avg[0] += avg_color[0]
    total_avg[1] += avg_color[1]
    total_avg[2] += avg_color[2]
    all.append(avg_color)
    print(avg_color)

print(total_avg[0] / length/4, total_avg[1] / length/4, total_avg[2] / length/4)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# reshape = image.reshape((image.shape[0] * image.shape[1], 3))
#
# # Find and display most dominant colors
# cluster = KMeans(n_clusters=5).fit(reshape)
# visualize = visualize_colors(cluster, cluster.cluster_centers_)
# visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
# cv2.imshow('visualize', visualize)
# cv2.waitKey()
