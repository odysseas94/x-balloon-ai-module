from datetime import datetime
from time import time, ctime
import cv2
from app.src.instances.Library import color_splash
from app.src.instances.LoadFromServer import LoadFromServer
import argparse
import os
import sys
import json
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils


class BalloonDataset(utils.Dataset):
    classification_models = {}
    shape_type_models = {}
    load_from_server = None
    MAX_INSTANCES_PER_IMAGE_TRAINING = 600
    file_name_set = "set.json"

    def __init__(self, load_from_server: LoadFromServer):
        super().__init__()
        self.load_from_server = load_from_server
        self.classification_models = load_from_server.classifications
        self.shape_type_models = load_from_server.shape_types
        self.prepare_classification_models()

    def prepare_classification_models(self):
        for key in self.classification_models:
            classification = self.classification_models[key]
            self.add_class("x-balloon", classification.id, classification.name)

    def load_dataset(self, dataset_dir, subset):
        # Train or validation dataset?
        assert subset in ["training", "validation"]
        dataset_dir = os.path.join(dataset_dir, subset)
        json_file = json.load(open(os.path.join(dataset_dir, "set.json")))
        annotations = list(json_file.values())
        # load regions
        annotations = [a for a in annotations if a['regions']]
        print(str(len(annotations)) + "  " + subset)

        # Add images
        for a in annotations:
            shapes = [r['shape_attributes'] for r in a['regions']]
            region_attributes = [s['region_attributes'] for s in a['regions']]
            ids = [int(n['id']) for n in region_attributes]
            # to reduce vram and ram
            if self.MAX_INSTANCES_PER_IMAGE_TRAINING and len(ids) > self.MAX_INSTANCES_PER_IMAGE_TRAINING:
                ids = ids[0:self.MAX_INSTANCES_PER_IMAGE_TRAINING]
                shapes = shapes[0:self.MAX_INSTANCES_PER_IMAGE_TRAINING]
            print("len : " + str(len(ids)), ids)

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            # ty= {"polygon":}

            self.add_image(
                "x-balloon",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                shapes=shapes,
                ids=ids
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]

        list_ids = info["ids"]

        print("" + ctime(time()) + " --> [" + str(len(list_ids)) + "] [" + str(info["path"]) + "]", list_ids)

        if info["source"] != "x-balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]

        mask = np.zeros([info["height"], info["width"], len(info["shapes"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["shapes"]):
            # Get indexes of pixels inside the polygon and set them to 1

            rr, cc = self.get_mask_by_shape_type(p)
            mask[rr, cc, i] = 1

        #self.show_single_mask(info, mask)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(list_ids, dtype=np.int32)

    def show_single_mask(self, info, mask):
        image = skimage.io.imread(info["path"])
        splash = color_splash(image, mask)
        splash = cv2.resize(splash, (1000, 1000))
        cv2.imshow('image', splash)
        cv2.waitKey(0)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "region":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def get_mask_by_shape_type(self, shape_attribute):
        if "name" not in shape_attribute:
            return None
        shape_name = shape_attribute["name"]
        # sk image uses all x,y inverse as y,x
        if shape_name == "rect":
            return self.rectangle(shape_attribute['y'], shape_attribute['x'], shape_attribute["width"],
                                  shape_attribute["height"])
        elif shape_name == "circle":
            return skimage.draw.circle(shape_attribute['cy'], shape_attribute['cx'], shape_attribute["r"])
        elif shape_name == "ellipse":
            return skimage.draw.ellipse(shape_attribute['cy'], shape_attribute['cx'], shape_attribute["ry"],
                                        shape_attribute["rx"])
        elif shape_name == "polygon" or shape_name == "multipolygon":
            return skimage.draw.polygon(shape_attribute['all_points_y'], shape_attribute['all_points_x'])

        return None

    @staticmethod
    def rectangle(r0, c0, width, height):
        print(r0, c0, width, height)
        rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]
        return skimage.draw.polygon(rr, cc)
