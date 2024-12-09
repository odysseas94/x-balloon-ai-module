from tkinter import Image

from imantics import Polygons

from app.src.connection.UploadMasksForMultipleImagesConnection import UploadMasksForMultipleImagesConnection
from app.src.models.ImageModel import ImageModel
from app.src.models.ShapeModel import ShapeModel


class GatherAllPolygons:
    image_models = None
    testing_id = None
    passing = None

    def __init__(self, image_models: [ImageModel], testing_id):
        self.image_models = {}
        self.passing = False
        for image in image_models:
            self.image_models[str(image.id)] = {}
            self.image_models[str(image.id)]["image"] = image.get_attributes()
        self.testing_id = testing_id
        self.image_models["testing_id"] = self.testing_id

    def push_polygon(self, image_model: ImageModel, polygons: [], class_ids=[], scores=[]):
        shapes = []
        self.passing = True
        for index in range(len(polygons)):
            polygon = polygons[index]
            class_id = class_ids[index]
            shape_model = ShapeModel(class_id=class_id)

            if shape_model.set_points(polygon):
                shapes.append(shape_model.get_attributes())
        self.image_models[str(image_model.id)]["shapes"] = shapes
        self.image_models[str(image_model.id)]["scores"] = scores

    def upload(self):
        if self.passing:
            UploadMasksForMultipleImagesConnection(self.image_models)
