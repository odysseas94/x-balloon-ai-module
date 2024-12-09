from datetime import datetime
import random

from app.src.models.DatabaseModel import DatabaseModel
from app.src.models.WeightFile import WeightFile
import time


class ShapeModel(DatabaseModel):
    id = 0
    points = None
    class_id = ""
    area = 0
    shape_type_id = None
    date_created = ""
    date_updated = ""

    def __init__(self, *args,
                 **kwargs):
        super().__init__("shape")
        self.id = kwargs.get('id', int(time.time() * 1000.0) + random.random() * 100000)
        self.points = kwargs.get('points', [])
        self.class_id = kwargs.get('class_id', 0)
        self.shape_type_id = kwargs.get('shape_type_id', 6)
        self.date_created = kwargs.get('date_created', int(time.time() * 1000.0))
        self.date_updated = kwargs.get('date_updated', None)

    def set_points(self, coordinates: []):
        index = 0
        while index < len(coordinates) - 1:
            self.points.append({"x": coordinates[index], "y": coordinates[index + 1]})
            index += 2
        self.area = self.calculate_polygon_area(self.points)
        if self.area > 0:
            return True
        return False

    def get_attributes(self):
        now = datetime.now()
        self.date_created = now.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "_id": self.id,
            "class_id": self.class_id,
            "automated": 1,
            "area": self.area,
            "shape_type_id": self.shape_type_id,
            "points": self.points,
            "date_created": self.date_created,
            # "date_updated": self.date_updated

        }

    def calculate_polygon_area(self, points):
        n = len(points)  # of points
        area = 0.0
        for i in range(n):
            j = (i + 1) % n

            area += points[i]["x"] * points[j]["y"]
            area -= points[j]["x"] * points[i]["y"]
        area = abs(area) / 2.0
        return area
