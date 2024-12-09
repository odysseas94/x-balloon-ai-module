from app.src.connection.UrlConnection import UrlConnection
from app.src.models.WeightFile import WeightFile


class UploadTrainingWeightsInfoConnection(UrlConnection):
    method = "POST"

    def __init__(self, training_id, weight: WeightFile):
        super().__init__("upload-training-weights")
        data = {"training_id": training_id, "weight": weight.get_attributes()}
        self.data = data
        self.init()
