
from app.src.generic.Socket import Socket


class SocketInstance(Socket):

    def __init__(self, application, user):
        super().__init__(application, user)

    def onBeginTesting(self, data):
        self.application.init("test")

    def onBeginTraining(self, data):
        self.application.init("train")

    def emitReadyToServe(self):
        self.io.emit("ready-serve")





