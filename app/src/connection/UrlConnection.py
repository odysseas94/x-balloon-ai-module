import json
import traceback

import requests
from requests.auth import HTTPBasicAuth
from collections import namedtuple


class UrlConnection:
    mainController = "ai/"
    domainUrl = "http://localhost:8888/index.php?r=api/v1/"
    username = "admin"
    password = "admin"
    url_appender = None
    method = "GET"
    url = None
    response = None

    data = None
    succeed = False
    json_data = None
    raw_data = None

    def __init__(self, url_appender: str = "get-url-attributes"):
        self.url_appender = url_appender
        self.url = self.domainUrl + self.mainController + self.url_appender

    def init(self):
        self.start()

    def success(self, data):
        pass

    def success_parent(self, data):
        self.json_data = data
        self.succeed = True
        self.success(data)

    def error(self, data):
        pass

    def start(self):
        if self.method == "GET":
            self.response = requests.get(self.url, verify=False,
                                         auth=HTTPBasicAuth(self.username, self.password), params=self.data)
        else:
            self.response = requests.post(self.url, verify=False,
                                          auth=HTTPBasicAuth(self.username, self.password), json=self.data)
        json_data = None
        try:

            print(self.url)
            # print(self.response.json())
            #   json_data = json.loads(self.response.text, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
            self.raw_data = json_data = self.response.json();

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return self.error("error parsing")

        if "error" in json_data:
            return self.error(json_data["error"])

        if "success" in json_data:
            return self.success_parent(json_data["success"])
        return self.error("error parsing")

        # j = json.loads(your_json)
        # u = User(**j)
