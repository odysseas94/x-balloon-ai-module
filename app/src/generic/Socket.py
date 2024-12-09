import socketio;
from app.src.instances.MachineHardware import MachineHardware

io = socketio.Client()


class Socket:
    url = "http://localhost:3001"

    def __init__(self, application, user):
        self.io = io
        self.user = user
        self.connected = False
        self.application = application
        self.machineHardware = MachineHardware()

    def init(self):
        print(self.user.token)
        self.io.connect(self.url,
                        {"type": "server", "token": self.user.token, "id": str(self.user.id),
                         "mac": MachineHardware.getMac()})
        self.bindEvents()
        self.io.wait()

    def onConnect(self):
        pass

    def _connect(self):
        self.connected = True
        self.onConnect()

    def onDisconnect(self):
        pass

    def _disconnect(self):
        self.connected = False
        self.onDisconnect()

    def onBeginTesting(self, data):
        pass

    def onBeginTraining(self, data):
        pass

    def onMachineHardware(self, data):
        self.emitMachineHardware()

    def emitMachineHardware(self):
        self.io.emit("machine-hardware", self.machineHardware.getAttributes())

    @staticmethod
    def printLog(*args, sep=' ', end='\n', file=None):
        result = []
        for arg in args:
            result.append(str(arg))

        if io.connected:
            io.emit("log-action", " ".join(result))
            print(*args, sep=' ', end='\n', file=None)

    def error(self):
        pass

    def bindEvents(self):
        self.io.on("connect", self._connect)
        self.io.on("error", self.error)
        self.io.on("disconnect", self._disconnect)
        self.io.on("machine-hardware", self.onMachineHardware)
        self.io.on("begin-training", self.onBeginTraining)
        self.io.on("begin-testing", self.onBeginTesting)
