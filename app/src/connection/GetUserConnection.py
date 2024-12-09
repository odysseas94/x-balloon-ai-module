from builtins import super
from app.src.connection.UrlConnection import UrlConnection
from app.src.models.UserModel import UserModel


class GetUserConnection(UrlConnection):
    mainController = "generic/"

    def __init__(self):
        super().__init__("get-user")
        self.user = None

        self.init()

    def success(self, data):
        json_data = data
        self.user = UserModel(**data)



    def error(self, data):
        print("error: " + str(data))
