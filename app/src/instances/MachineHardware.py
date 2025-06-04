import os
import platform
import re
import uuid
import psutil


class MachineHardware:
    cpu_cores = 0
    cpu_name = ""
    gpus = []
    name = ""
    ram = 0
    os = ""
    inited = False

    def __init__(self):
        self.inited = False

    def init(self):
        self.machineCore()
        #self.gpuInfo()
        self.inited = True
        return self

    def machineCore(self):
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.ram = psutil.virtual_memory().total
        self.cpu_name = platform.processor()
        self.name = platform.node()
        self.os = platform.platform()

    def gpuInfo(self):
        gpus_array = nvgpu.gpu_info()
        for gpu in gpus_array:
            self.gpus.append({
                "name": gpu["type"],
                "vram": gpu["mem_total"] * pow(1024, 2),
                "uuid": gpu["uuid"]
            })

    def getAttributes(self):
        if not self.inited:
            self.init()
        return {
            "name": self.name,
            "mac": self.getMac(),
            "os": self.os,
            "cpu": {
                "name": self.cpu_name,
                "cores": self.cpu_cores
            },
            "ram": self.ram,
            "gpus": self.gpus
        }

    @staticmethod
    def getMac():
        mac = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
        return mac
