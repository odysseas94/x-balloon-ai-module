import json

from app.src.connection.UrlConnection import UrlConnection


class UploadMasksForMultipleImagesConnection(UrlConnection):
    method = "POST"

    def __init__(self, data):
        super().__init__("upload-masks")
        # print(json.dumps(data))
        self.data = data

        self.init()

    def success(self, data):
        # print(data)
        json_data = data

    def error(self, data):
        print("error: " + str(data))
