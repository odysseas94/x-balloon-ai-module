from app.src.export.ContainerImage import ContainerImage
from app.src.library.KMeansCluster import KMeansCluster
from app.src.models.ImageModel import ImageModel
import numpy as np
import skimage
import cv2
import skimage.io
import skimage.color
from PIL import ImageColor


class ImagePainter:
    image_path: str = None
    image_model: ImageModel = None
    container_image: ContainerImage = None
    classification_models: {}
    cpa_area_by_classification: []
    k_means_cluster: KMeansCluster = None
    image_name_tosave = ""

    def __init__(self, container_image: ContainerImage, classification_models, image_name_tosave="training"):
        self.container_image = container_image
        self.image_path = container_image.resized_image_path
        self.image_name_tosave = image_name_tosave
        self.image_model = container_image.image_model
        self.classification_models = classification_models
        self.cpa_area_by_classification = []
        self.k_means_cluster = self.container_image.k_means_cluster
        self.draw_image()
        self.draw_masks_per_class()

    def draw_image(self):
        mask = np.asarray(self.container_image.actual_image);

        for i, shape in enumerate(self.image_model.shapes):
            rr, cc = shape.get_mask_by_shape_type()
            rgb = [i for i in ImageColor.getrgb(self.classification_models[shape.class_id].color)];
            mask[rr, cc, :] = [rgb[2], rgb[1], rgb[0]]
        cv2.imwrite(self.container_image.export_path + "/" + self.image_name_tosave + self.image_model.extension, mask)

    def draw(self):
        mask = np.zeros([self.container_image.resized_image_height, self.container_image.resized_image_width,
                         len(self.image_model.shapes)],
                        dtype=np.uint8)
        for i, shape in enumerate(self.image_model.shapes):
            rr, cc = shape.get_mask_by_shape_type()

            mask[rr, cc, i] = 1
        if len(self.image_model.shapes) > 0:
            self.show_single_mask(mask)

    def show_single_mask(self, mask):
        image = skimage.io.imread(self.image_path)
        splash = color_splash(image, mask)
        splash = cv2.resize(splash,
                            (self.container_image.resized_image_width, self.container_image.resized_image_height))
        cv2.imshow('image', splash)
        cv2.waitKey(0)

    def draw_masks_per_class(self):

        for key, value in self.classification_models.items():
            mask = np.zeros([self.container_image.resized_image_height, self.container_image.resized_image_width],
                            dtype=np.uint8)
            for i, shape in enumerate(self.image_model.shapes):
                if key == shape.class_id:
                    rr, cc = shape.get_mask_by_shape_type()
                    mask[rr, cc] = 1

            logical = np.logical_and(mask, self.k_means_cluster.output_binary_image)
            logical_count = logical[logical > 0].shape[0]
            # self.save_mask_for_single_annotation(mask, "shape_class_id" + str(key))
            self.cpa_area_by_classification.append(logical_count)

    def save_mask_for_single_annotation(self, mask, image_name):
        mask = mask * 255
        cv2.imwrite(self.container_image.export_path + "/"  + image_name + ".jpg",
                    mask)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 150
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)

    if mask.shape[0] > 0:

        splash = np.where(mask, image, gray).astype(np.uint8)

    else:
        splash = gray
    return splash
