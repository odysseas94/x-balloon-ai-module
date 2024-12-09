import os

from app.src.instances.ShapeInstance import ShapeInstance
from app.src.models.ImageModel import ImageModel
from urllib import request
from os import path
import json
from os import listdir
from os.path import isfile, join


class SetInstance:
    image_models = None
    set_json = []
    type = "validation",
    dataset_model = None
    current_path = ""
    image_names = None
    global_path = None
    saved_path = None

    def __init__(self, type_set: str, dataset_model, path_dataset: str, set_json):
        self.set_json = set_json
        self.type = type_set
        self.path = path_dataset;
        self.global_path = path
        self.dataset_model = dataset_model
        self.current_path = path_dataset + "/" + str(self.type) + "/"
        self.image_names = []
        self.image_models = []
        self.image_names = [f for f in listdir(self.current_path) if isfile(join(self.current_path, f))]

        self.init_images()
        self.download_images()
        self.save_json_set()

    def init_images(self):

        for index, key in enumerate(self.set_json):
            single_file = self.set_json[key]
            if "image_attributes" in single_file:
                current_image = ImageModel(**single_file["image_attributes"])
                self.image_models.append(current_image)
                if "regions" in single_file:
                    regions = single_file["regions"]
                    print(" regions")
                    for inner_region in regions:
                        shape_instance = None
                        if "shape_attributes" in inner_region:
                            shape_attributes = inner_region["shape_attributes"];
                            if not (shape_attributes["name"]):
                                continue
                            shape_instance = ShapeInstance(shape_attributes["name"])

                            shape_instance.all_points_x = self.key_exists("all_points_x", shape_attributes)
                            shape_instance.all_points_y = self.key_exists("all_points_y", shape_attributes)
                            shape_instance.x = self.key_exists("x", shape_attributes)
                            shape_instance.rx = self.key_exists("rx", shape_attributes)
                            shape_instance.cx = self.key_exists("cx", shape_attributes)
                            shape_instance.y = self.key_exists("y", shape_attributes)
                            shape_instance.ry = self.key_exists("ry", shape_attributes)
                            shape_instance.cy = self.key_exists("cy", shape_attributes)
                            shape_instance.r = self.key_exists("r", shape_attributes)
                            shape_instance.height = self.key_exists("height", shape_attributes)
                            shape_instance.height = self.key_exists("width", shape_attributes)
                            if "region_attributes" in inner_region and isinstance(shape_instance, ShapeInstance):
                                region_attributes = inner_region["region_attributes"]
                                shape_instance.class_id = self.key_exists("id", region_attributes)
                                shape_instance.class_name = self.key_exists("name", region_attributes)

                        if isinstance(shape_instance, ShapeInstance):
                            current_image.shapes.append(shape_instance)

            print(self.type + " ----> " + str(current_image.__str__()))

    def download_images(self):
        for model in self.image_models:

            if not path.exists(self.current_path + model.canonical_name):
                try:
                    file = open(self.current_path + model.canonical_name, 'wb')
                    print(model.image)
                    file.write(request.urlopen(model.image).read())
                    file.close()
                except Exception as e:
                    print(e)
            if model.canonical_name in self.image_names:
                self.image_names.remove(model.canonical_name)

        # delete other

        for name in self.image_names:
            if os.path.exists(self.current_path + name) and "set.json" not in name:
                os.remove(self.current_path + name)

    def key_exists(self, key, array):
        if key in array:
            return array[key]
        else:
            return None

    def save_json_set(self):
        with open(self.current_path + "set.json", 'w') as f:
            json.dump(self.set_json, f, ensure_ascii=False)
