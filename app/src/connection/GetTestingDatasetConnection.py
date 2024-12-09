from builtins import super
from app.src.connection.UrlConnection import UrlConnection


class GetTestingDatasetConnection(UrlConnection):
    #   mainUrl = "http://localhost:8888/index.php?r=api/v1/ai/"

    def __init__(self):
        super().__init__("get-testing-set")
        self.init()

    def success(self, data):
        json_data = data

    def error(self, data):
        print("error: " + str(data))
