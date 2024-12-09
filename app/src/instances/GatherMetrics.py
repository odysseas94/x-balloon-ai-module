import json

from app.src.config.AIConfig import AiConfig
from app.src.connection.UploadTrainingWeightsInfoConnection import UploadTrainingWeightsInfoConnection
from app.src.models.WeightFile import WeightFile


class GatherMetrics:
    metrics = None
    single_metrics = {}
    weight_file_model = None
    configuration = None
    weight_file_model = None
    training_id = 0
    child_weight = None

    def __init__(self, configuration: AiConfig, metrics: [], training_id, child: WeightFile):
        self.metrics = metrics
        self.configuration = configuration
        self.training_id = training_id
        self.child_weight = child;
        self.make_single_metrics()
        self.weight_file_model = WeightFile(**self.single_metrics)

    def make_single_metrics(self):
        key = list(self.metrics.keys())[-1]
        self.single_metrics = self.metrics[key]

        self.single_metrics["configuration"] = self.configuration.get_info()
        if self.child_weight:
            self.single_metrics["id"] = self.child_weight.id
        else:
            self.single_metrics["id"] = None
        self.single_metrics["name"] = "mask_rcnn_x-balloon_{:04d}.h5".format(int(key))
        self.single_metrics["path"] = str(self.training_id) + "/mask_rcnn_x-balloon_{:04d}.h5".format(int(key))
        self.single_metrics["success_ratio"] = 100 - self.single_metrics["loss"]
        self.single_metrics["error_ratio"] = self.single_metrics["loss"]
        self.single_metrics["date_created"] = None
        self.single_metrics["date_updated"] = None

        print(self.single_metrics)

        return self.single_metrics

    def upload(self):
        UploadTrainingWeightsInfoConnection(self.training_id, self.weight_file_model)
