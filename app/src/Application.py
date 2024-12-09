import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from app.src.export.ExportHandler import ExportHandler
from app.src.instances.ShapeInstance import ShapeInstance
from app.src.models.ImageModel import ImageModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import traceback

import imgaug
import keras
from imantics import Mask
from tensorflow import InteractiveSession

from app.src.config.AIConfig import AiConfig, InferenceAiConfig
from app.src.config.BalloonDataset import BalloonDataset
from app.src.config.CustomCallbacks import MetricsCallback
from app.src.connection.BeginTestingConnection import BeginTestingConnection
from app.src.connection.CompleteTestingConnection import CompleteTestingConnection
from app.src.instances.Augmentation import Augmentation
from app.src.instances.GatherAllPolgyons import GatherAllPolygons
from app.src.instances.GatherMetrics import GatherMetrics
from app.src.instances.Library import color_splash
from app.src.instances.LoadFromServer import LoadFromServer
import argparse

import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf

from app.src.instances.SocketInstance import SocketInstance

printLog = SocketInstance.printLog
from mrcnn import model as modellib, utils, visualize

import datetime

trainedReformed = None


class Application:
    model = None
    load_from_server = None
    dataset_train = None
    dataset_val = None
    layers = "all"
    dataset_empty = None
    mode = "training"
    action = "train"
    metricsCallback = None
    checkPointCallback = None
    socket = None
    export_handler = None

    def __init__(self):
        self.action = "train"

    def setSocket(self, socket: SocketInstance):
        self.socket = socket

    def endedAction(self):
        if self.socket:
            self.socket.emitReadyToServe()

    def init(self, action="train"):
        try:
            if action == "train":
                self.load_from_server = LoadFromServer()
                self.config = AiConfig(self.load_from_server)
                return;
            else:
                self.action = "testing"
                self.mode = "inference"
                self.set_growth_gpu_testing()
                self.load_from_server = LoadFromServer("testing")
                self.config = InferenceAiConfig(self.load_from_server)

            # set configuration
            if not (self.load_from_server.data_testing or self.load_from_server.data_dataset):
                printLog("Set is empty")
                self.endedAction()
                return 0

            self.config.display()
            # load our model
            self.load_model()

            if action == "train":
                self.metricsCallback = MetricsCallback(self.load_from_server)
                self.checkPointCallback = keras.callbacks.ModelCheckpoint(
                    self.load_from_server.dataset_path + "/mask_rcnn_x-balloon_" + "{epoch:04d}.h5",
                    verbose=1, period=50, save_weights_only=True)
                self.train()

            else:

                self.detect_and_color_splash()

        except Exception as e:
            traceback.print_exc()
            printLog("Error", e)
            self.endedAction()

    def train(self):
        """Train the model."""
        # Training dataset.
        printLog("Starting Training")
        printLog("Config", self.config.get_info())
        printLog("Preparing Training Set")
        self.dataset_train = BalloonDataset(self.load_from_server)
        self.dataset_train.load_dataset(self.load_from_server.dataset_path, "training")
        self.dataset_train.prepare()

        printLog("Preparing Validation Set")
        # Validation dataset
        self.dataset_val = BalloonDataset(self.load_from_server)
        self.dataset_val.load_dataset(self.load_from_server.dataset_path, "validation")
        self.dataset_val.prepare()

        self.model.train(self.dataset_train, self.dataset_val,
                         learning_rate=self.config.LEARNING_RATE,
                         # callbacks
                         custom_callbacks=[self.checkPointCallback, self.metricsCallback],
                         epochs=self.config.MAX_EPOCHS,
                         layers=self.layers, augmentation=Augmentation.get_full())

        self.model.train(self.dataset_train, self.dataset_val,
                         learning_rate=self.config.LEARNING_RATE / 10,
                         # callbacks
                         custom_callbacks=[self.checkPointCallback, self.metricsCallback],
                         epochs=self.config.MAX_EPOCHS * 2,
                         layers="all", augmentation=Augmentation.get_full())
        printLog("Training Ended")

        self.load_from_server.write_down_json_metrics(self.metricsCallback.metrics)
        model_path = os.path.join(self.load_from_server.dataset_log,
                                  "mask_rcnn_x-balloning-final.h5")
        self.model.keras_model.save_weights(model_path)
        printLog("Uploading all metrics to server")
        # save weight to server
        gather_metrics = GatherMetrics(self.config, self.metricsCallback.metrics,
                                       self.load_from_server.datasetModel.training_id,
                                       self.load_from_server.datasetModel.weight_child)
        gather_metrics.upload()
        self.endedAction()

    def detect_and_color_splash(self):

        self.export_handler = ExportHandler(self.load_from_server)

        printLog("Config", self.config.get_info())
        printLog("Starting Testing")
        printLog("Downloading all images from server")
        BeginTestingConnection(self.load_from_server.testingModel.id)
        testing_set = self.load_from_server.testingInstance
        self.dataset_empty = BalloonDataset(self.load_from_server)
        # self.dataset_empty.load_dataset(self.load_from_server.dataset_path, "testing")
        self.dataset_empty.prepare()

        # Read image
        images = []
        printLog("Images Length :", len(testing_set.image_models))
        for image in testing_set.image_models:
            images.append(skimage.io.imread(testing_set.current_path + image.canonical_name))

            # Detect objects

        for index_images in range(len(images)):

            image_model = testing_set.image_models[index_images]
            printLog("Testing Image" + image_model.canonical_name)
            results = self.model.detect([images[index_images]], verbose=1)

            for index_result in range(len(results)):
                result = results[index_result]
                masks = result["masks"]
                scores = result["scores"]
                class_ids = result["class_ids"]

                collector = GatherAllPolygons([image_model], self.load_from_server.testingModel.id)
                splash = color_splash(images[index_images], masks)
                printLog(masks.shape[2], "Masks")
                polygons = []
                classifications_ids = []
                score_lists = []

                for c in range(masks.shape[2]):
                    polygon = Mask(masks[:, :, c]).polygons()

                    if len(polygon.segmentation):
                        # if one one mask there are multiple shapes
                        for index_polygons in range(len(polygon.segmentation)):
                            polygons.append(polygon.segmentation[index_polygons])
                            classifications_ids.append(str(class_ids[c]))
                            score_lists.append(str(scores[c]))

                # print(score_lists)
                # To export annotatioons
                self.export_shapes(image_model, polygons, classifications_ids)
                collector.push_polygon(image_model, polygons, classifications_ids, score_lists)
                length = len(image_model.canonical_name)
                new_name = image_model.canonical_name[0:length - 4] + "-classified" + image_model.canonical_name[
                                                                                      length - 4:length]
                printLog("Saving Image to localhost and uploading data and masks to server")
                # save image
                skimage.io.imsave(testing_set.current_path + new_name, splash)
                self.export_handler.save_excel_export()
                #collector.upload()
        printLog("Testing Ended")
        #CompleteTestingConnection(self.load_from_server.testingModel.id)
        self.endedAction()


    def export_shapes(self, image: ImageModel, polygons: [], class_ids):

        index = 0
        for polygon in polygons:
            shape_instance = ShapeInstance("multipolygon")

            shape_instance.canonical_points_to_all_points(polygon)
            shape_instance.class_id = int(class_ids[index])
            image.shapes.append(shape_instance)
            index += 1
        self.export_handler.init_instances_testing_image(image)

    def load_model(self):
        self.model = modellib.MaskRCNN(mode=self.mode, config=self.config,
                                       model_dir=self.load_from_server.dataset_log)

        # remove unnecessary weights
        exclude = []
        if "mask_rcnn_coco.h5" in self.load_from_server.training_dataset_path:
            exclude = [
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"]
            # if not old dataset then train old weight file + current data
            self.layers = "heads"
        print("layers" + self.layers)

        if self.action == "train":
            self.model.load_weights(self.load_from_server.training_dataset_path, by_name=True, exclude=exclude)
        else:
            self.model.load_weights(self.load_from_server.testing_dataset_path, by_name=True, exclude=exclude)

    @staticmethod
    def set_growth_gpu_testing():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        keras.backend.set_session(session)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train or detect')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    args = parser.parse_args()
    print(args.command)

    trainedReformed = Application()
    trainedReformed.init(args.command)
