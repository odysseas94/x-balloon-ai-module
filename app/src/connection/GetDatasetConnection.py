from builtins import super

from app.src.connection.UrlConnection import UrlConnection


class GetDatasetConnection(UrlConnection):
    def __init__(self):
        super().__init__("get-full-dataset")
        self.init()

    def success(self, data):
        json_data = data

    def error(self, data):
        print("error: " + str(data))
