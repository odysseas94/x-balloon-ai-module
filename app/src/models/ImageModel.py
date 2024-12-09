import os

from app.src.models.DatabaseModel import DatabaseModel
from app.src.models.WeightFile import WeightFile
from os.path import splitext



class ImageModel(DatabaseModel):
    id = 0
    name = ""
    path = ""
    size = ""
    image = ""  # url
    thumbnail = ""
    thumbnail_path = ""
    date_created = ""
    date_updated = ""
    canonical_name = ""
    extension = ""
    shapes = []

    def __init__(self, id, name, path, size, image, thumbnail_path, thumbnail, date_created, date_updated, *args,
                 **kwarg):
        super().__init__("image")
        self.id = id
        self.name = name
        self.path = path
        self.size = size
        self.image = image
        self.shapes = []
        self.thumbnail = thumbnail
        self.thumbnail_path = thumbnail_path
        self.date_created = date_created
        self.date_updated = date_updated
        _, self.canonical_name = os.path.split(path)
        self.extension = splitext(self.canonical_name)[1]

    def get_attributes(self):
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "thumbnail_path": self.thumbnail_path,
            "size": self.size,
            "date_created": self.date_created,
            "date_updated": self.date_updated

        }

    def scale_all_shapes(self, scale):
        for shape in self.shapes:
            shape.start_scaling(scale)

    def __eq__(self, obj):
        if isinstance(obj, self.__class__):
            return self.id == obj.id
        return False

    def __str__(self):
        return str(self.id) + "  " + self.canonical_name + ", shapes: " + str(len(self.shapes))
