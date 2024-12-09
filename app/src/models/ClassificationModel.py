from app.src.models.DatabaseModel import DatabaseModel
from app.src.models.WeightFile import WeightFile


class ClassificationModel(DatabaseModel):
    id = 0
    name = ""
    pretty_name = ""
    date_created = ""
    date_updated = ""
    color = ""

    def __init__(self, id, name, pretty_name,color, date_created, date_updated, *args,
                 **kwargs):
        super().__init__("classification")
        self.id = id
        self.name = name
        self.color = color
        self.pretty_name = pretty_name
        self.date_created = date_created
        self.date_updated = date_updated
