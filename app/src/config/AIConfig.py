from mrcnn.config import Config
from app.src.instances.LoadFromServer import LoadFromServer
import numpy as np


class AiConfig(Config):
    NAME = "IBD"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.resnet30
    IMAGES_PER_GPU = 1
    LEARNING_RATE = 0.001
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1  # Background + toy
    POST_NMS_ROIS_TRAINING = 10000
    POST_NMS_ROIS_INFERENCE = 10000
    # Number of training steps per epoch
    MAX_EPOCHS = 200
    STEPS_PER_EPOCH = 220
    VALIDATION_STEPS = 50
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 1
    TRAIN_ROIS_PER_IMAGE = 500
    MAX_GT_INSTANCES = 100
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 512  # 300
    # Max number of final detections per image
    # DETECTION_MAX_INSTANCES = 512
    # 50 hidden layers
    BACKBONE = "resnet101"
    IMAGE_RESIZE_MODE = "square"
    MEAN_PIXEL = np.array([57.22880641084562, 57.804680468324506, 58.82812760203907])
    # i should try also 16,32,64,128,256
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5
    # set it as lower as u can.
    # its more flexible at high and at low to learn strictly what u want
    # more propsals
    RPN_NMS_THRESHOLD = 0.99
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    load_from_server = None

    def __init__(self, load_from_server: LoadFromServer):
        self.load_from_server = load_from_server
        self.NUM_CLASSES += len(load_from_server.classifications)
        super().__init__()

    def get_info(self):
        _str = "{"
        for a in dir(self):
            if not a.startswith("__") and not a.islower() and not callable(getattr(self, a)):
                _str += ('"{}":"{}", '.format(a, getattr(self, a)))
        _str += "}"

        return _str


class InferenceAiConfig(AiConfig):
    IMAGES_PER_GPU = 1

    GPU_COUNT = 1
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.5    # set it as lower as u can.
    # its more flexible at high and at low to learn strictly what u want
    RPN_NMS_THRESHOLD = 0.99
    TRAIN_ROIS_PER_IMAGE = 2048 * 2
    MAX_GT_INSTANCES = 2048 * 2
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_SCALE = 1
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    RPN_TRAIN_ANCHORS_PER_IMAGE = 2048 * 2  # 300
    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 2048 * 2

    def __init__(self, load_from_server: LoadFromServer):
        super().__init__(load_from_server)
