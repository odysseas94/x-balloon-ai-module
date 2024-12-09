import argparse
import os
import sys

from app.src.config.AIConfig import AiConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from imantics import Polygons, Mask
import cv2
from tensorflow import InteractiveSession

from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import pickle;
import tensorflow as tf
# Root directory of the project
from app.src.instances.LoadFromServer import LoadFromServer

ROOT_DIR = os.path.abspath("../../")
THIS_DIR = os.path.abspath("")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

WEIGHT_PATH = "../datasets/1/logs/x-balloon20200629T1342/mask_rcnn_x-balloon_0040.h5"
IMAGE_PATH = '../datasets/1/testing/1589788879517517010.jpg'


class CustomConfig(Config):
    NAME = "x-balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.resnet30
    IMAGES_PER_GPU = 1
    LEARNING_RATE = 0.001
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + toy
    POST_NMS_ROIS_TRAINING = 6000
    POST_NMS_ROIS_INFERENCE = 2000
    # Number of training steps per epoch
    MAX_EPOCHS = 200
    STEPS_PER_EPOCH = 250
    VALIDATION_STEPS = 50
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 0
    TRAIN_ROIS_PER_IMAGE = 512
    MAX_GT_INSTANCES = 512
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512  # 300
    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 512
    # 50 hidden layers
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "crop"

    MEAN_PIXEL = np.array([226.78973934096723, 207.18995435393035, 232.21716953829852])
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Skip detections with < 40% confidence
    DETECTION_MIN_CONFIDENCE = 0.5
    # set it as lower as u can.
    # its more flexible at high and at low to learn strictly what u want
    RPN_NMS_THRESHOLD = 0.9


class InferenceConfig(CustomConfig):
    IMAGES_PER_GPU = 1

    GPU_COUNT = 1
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.5
    # set it as lower as u can.its more flexible and high to learn strictly what u want
    RPN_NMS_THRESHOLD = 0.1
    TRAIN_ROIS_PER_IMAGE = 2048
    MAX_GT_INSTANCES = 2048
    IMAGE_RESIZE_MODE = "pad64"
    RPN_TRAIN_ANCHORS_PER_IMAGE = 2048  # 300
    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 2048


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("type", 1, "ballooning")
        self.add_class("type", 2, "inflammation")
        self.add_class("type", 3, "fat")
        self.add_class("type", 4, "sinusoid")
        self.add_class("type", 5, "vein")
        # Train or validation dataset?
        assert subset in ["training", "validation"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        print(os.path.join(dataset_dir, "set.json"))
        annotations1 = json.load(open(os.path.join(dataset_dir, "set.json")))
        print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            shapes = [r['shape_attributes'] for r in a['regions']]
            region_attributes = [s['region_attributes'] for s in a['regions']]
            ids = [int(n['id']) for n in region_attributes]
            print(ids)

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            # ty= {"polygon":}

            self.add_image(
                "type",  ## for a single class just add the name here
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

        # If not a bottle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        list_ids = image_info["ids"]
        print("gonna load masks for image" + str(image_id), list_ids)
        # print("ids")

        # if image_info["source"] != "region":
        #     return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["shapes"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["shapes"]):
            # Get indexes of pixels inside the polygon and set them to 1

            print(p["name"]);

            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1
            # image = skimage.io.imread(info["path"])
            # splash = color_splash(image, mask)
            # splash = cv2.resize(splash, (1000, 1000))
            # cv2.imshow('image', splash)
            # cv2.waitKey(0)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(list_ids)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "region":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(loadFromServer.dataset_path, "training")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(loadFromServer.dataset_path, "validation")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')
    print("gonna save ")

    model_path = os.path.join(THIS_DIR, "mask_rcnn_x-balloning" + str(datetime.datetime.now().timestamp()) + ".h5")
    model.keras_model.save_weights(model_path)


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
    print(mask.shape)
    # Copy color pixels from the original color image where mask is set
    print(mask.shape)
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect

        dataset_train = CustomDataset()
        dataset_train.load_custom(loadFromServer.dataset_path, "training")
        dataset_train.prepare()
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        masks = r['masks'];

        print("total found : ", masks.shape[2])
        # visualize.display_instances(
        #     image, r['rois'], r['masks'], r['class_ids'],
        #     dataset_train.class_names, r['scores'],
        #     show_bbox=False, show_mask=True,
        #     title="Predictions")
        # Save output

        print(r['class_ids'], r['scores'], print(masks.shape))

        date = datetime.datetime.now()
        file_name = "../results/splash_{:%Y%m%dT%H%M%S}.png".format(date)
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train or detect')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    args = parser.parse_args()
    loadFromServer = None
    mode = "training"
    weights_path = ""

    # Configurations
    if args.command == "train":
        config = CustomConfig()
        loadFromServer = LoadFromServer()
        weights_path = loadFromServer.training_dataset_path

    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        config = InferenceConfig()
        mode = "inference"
        loadFromServer = LoadFromServer("testing")
        weights_path = WEIGHT_PATH
    config.display()

    # Create model
    print("dataset ", loadFromServer)

    model = modellib.MaskRCNN(mode=mode, config=config,
                              model_dir=loadFromServer.dataset_path)

    # Load weights
    # sys.exit(0)

    model.load_weights(weights_path, by_name=True)

    # model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        detect_and_color_splash(model,
                                image_path=IMAGE_PATH)
