from app.src.instances.SetInstance import SetInstance
from app.src.instances.LoadFromServer import LoadFromServer
from app.src.models.UserModel import UserModel


class SetInstancesExport:
    load_from_server_training: LoadFromServer = None
    load_from_server_testing: LoadFromServer = None
    validation_instance: SetInstance = None
    training_instance: SetInstance = None
    testing_instance: SetInstance = None

    def __init__(self, load_from_server_training: LoadFromServer, load_from_server_testing: LoadFromServer):
        self.load_from_server_training = load_from_server_training
        self.load_from_server_testing = load_from_server_testing
        self.validation_instance = self.load_from_server_training.validationInstance
        self.training_instance = self.load_from_server_training.trainingInstance
        self.testing_instance = self.load_from_server_testing.testingInstance
