from builtins import super
from app.src.connection.UrlConnection import UrlConnection


class CompleteTestingConnection(UrlConnection):

    def __init__(self, testing_id):
        super().__init__("complete-testing")
        self.data = {"testing_id": testing_id}
        self.init()

    def success(self, data):
        json_data = data

    def error(self, data):
        print("error: " + str(data))
