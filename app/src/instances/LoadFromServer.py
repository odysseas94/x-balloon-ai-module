import json

from app.src.connection.GetDatasetConnection import GetDatasetConnection
from app.src.models.DatasetModel import DatasetModel
from pathlib import Path
import os

from app.src.connection.GetAllEssentialsConnection import GetAllEssentialsConnection
from app.src.connection.GetTestingDatasetConnection import GetTestingDatasetConnection
from app.src.instances.SetInstance import SetInstance
from app.src.models.ClassificationModel import ClassificationModel
from app.src.models.ShapeTypeModel import ShapeTypeModel

# outside the src (src siblings)
from app.src.models.TestingModel import TestingModel

GLOBAL_PATH = os.path.abspath(__file__ + "../../../../")
print(GLOBAL_PATH, "GLOBAL PATH")
DATASET_PATH = GLOBAL_PATH + "/datasets"


class LoadFromServer:
    datasetModel = None
    dataset_path = DATASET_PATH
    dataset_log = DATASET_PATH
    training_dataset_path = ""
    testing_dataset_path = ""
    data_dataset = None
    trainingInstance = None
    training_json = None
    testingInstance = None
    testingModel = None
    validationInstance = None
    validation_json = None
    essentialsConnection = None
    data_essentials = None
    datasetTrainingValidationConnection = None
    testingConnection = None
    data_testing = None

    _type = "training-validation"
    # map
    classifications = {}
    # map
    shape_types = {}

    def __init__(self, _type="training-validation"):
        if _type == "training-validation":
            print(self._type)
            self.datasetTrainingValidationConnection = GetDatasetConnection()
            self.parse_json_dataset()
        else:
            self._type = "testing"
            print(self._type)
            self.testingConnection = GetTestingDatasetConnection()
            self.parse_json_testing()

        self.essentialsConnection = GetAllEssentialsConnection()
        self.parse_json_essentials()

    def parse_json_testing(self):
        if not self.testingConnection.succeed or not self.testingConnection.json_data:
            print("error")
            return
        self.data_testing = self.testingConnection.json_data
        self.parse_dataset(self.data_testing)
        self.dataset_path += "/" + str(self.datasetModel.id)
        self.dataset_log = self.dataset_path + "/logs"
        if self.datasetModel.weight_child:
            self.testing_dataset_path = DATASET_PATH + "/" + self.datasetModel.weight_child.path
        print("testing_dataset_path", self.testing_dataset_path)
        self.create_dataset_folder()
        self.parse_testing()

    # check the json form connection for testing and validation
    def parse_json_dataset(self):
        if not self.datasetTrainingValidationConnection.succeed:
            print("error")
            return

        self.data_dataset = self.datasetTrainingValidationConnection.json_data
        self.parse_dataset(self.data_dataset)
        self.dataset_path += "/" + str(self.datasetModel.id)
        self.dataset_log = self.dataset_path + "/logs"
        self.training_dataset_path = DATASET_PATH + "/" + self.datasetModel.weight_parent.path
        self.create_dataset_folder()
        self.parse_training()
        self.parse_validation()

    # parse the object dataset
    def parse_dataset(self, json_data):
        data = json_data
        if "dataset" in data:
            dataset = data["dataset"]
            self.datasetModel = DatasetModel(**dataset)
            print(self.datasetModel.weight_parent.path)

    def parse_testing(self):
        if "testing_images" in self.data_testing:
            print("testing")
            self.testing_json = self.data_testing["testing_images"]
            self.testingInstance = SetInstance("testing", self.datasetModel, self.dataset_path, self.testing_json)
        if "testing" in self.data_testing:
            self.testingModel = TestingModel(**self.data_testing["testing"])

    # parse training data
    def parse_training(self):
        if "training" in self.data_dataset:
            print("training")
            self.training_json = self.data_dataset["training"]
            self.trainingInstance = SetInstance("training", self.datasetModel, self.dataset_path, self.training_json)

    # parse validation data
    def parse_validation(self):
        if "validation" in self.data_dataset:
            print("validation")
            self.validation_json = self.data_dataset["validation"]
            self.validationInstance = SetInstance("validation", self.datasetModel, self.dataset_path,
                                                  self.validation_json)

    # create all folder needed for the training/validation/testing
    def create_dataset_folder(self):
        # create if doesn't exists otherwise dont toych
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
        Path(self.dataset_path + "/validation").mkdir(parents=True, exist_ok=True)
        Path(self.dataset_path + "/training").mkdir(parents=True, exist_ok=True)
        Path(self.dataset_path + "/logs").mkdir(parents=True, exist_ok=True)
        Path(self.dataset_path + "/testing").mkdir(parents=True, exist_ok=True)
        if self.datasetTrainingValidationConnection:
            with open(self.dataset_path + "/response.json", 'w') as f:
                json.dump(self.datasetTrainingValidationConnection.raw_data, f, ensure_ascii=False)
        elif self.testingConnection:
            with open(self.dataset_path + "/testing.json", 'w') as f:
                json.dump(self.testingConnection.raw_data, f, ensure_ascii=False)

    # parse the json from the connection to essentials of (classes,shape types)

    def parse_json_essentials(self):
        if not self.essentialsConnection.succeed:
            print("error")
            return

        self.data_essentials = self.essentialsConnection.json_data
        print(self.data_essentials)
        if "classifications" in self.data_essentials:
            for classification in self.data_essentials['classifications']:
                model = ClassificationModel(**classification)
                self.classifications[model.id] = model

        if "shape_types" in self.data_essentials:
            for shapeType in self.data_essentials['shape_types']:
                model = ShapeTypeModel(**shapeType)
                self.shape_types[model.id] = model
        with open(self.dataset_path + "/essentials.json", 'w') as f:
            json.dump(self.essentialsConnection.raw_data, f, ensure_ascii=False)

    def write_down_json_metrics(self, metrics: []):
        with open(self.dataset_path + "/metrics.json", 'w') as f:
            json.dump(metrics, f, ensure_ascii=False)

    def read_metrics_file(self):
        return json.load(open(self.dataset_path + "/metrics.json"))

    def dump(self):
        for attr in dir(self):
            print("obj.%s = %r" % (attr, getattr(self, attr)))

# LoadFromServer()
