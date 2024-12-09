from app.src.Application import Application
from app.src.connection.GetUserConnection import GetUserConnection
from app.src.instances.SocketInstance import SocketInstance

application = Application()

userConnection = GetUserConnection()
socketInstance = SocketInstance(application, userConnection.user)
application.setSocket(socketInstance)
socketInstance.init()
