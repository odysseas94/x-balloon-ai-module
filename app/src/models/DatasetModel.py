from app.src.models.DatabaseModel import DatabaseModel
from app.src.models.WeightFile import WeightFile


class DatasetModel(DatabaseModel):
    id = 0
    weight_child = ""
    weight_parent = ""
    date_created = ""
    date_updated = ""

    def __init__(self, id, training_id, validation_id, weight_parent, date_created, date_updated, *args,
                 **kwargs):
        super().__init__("dataset")
        self.id = id
        self.training_id = training_id
        self.validation_id = validation_id;

        if "weight_child" in kwargs:
            self.weight_child = WeightFile(**kwargs["weight_child"])
        if weight_parent:
            self.weight_parent = WeightFile(**weight_parent)

        self.date_created = date_created
        self.date_updated = date_updated
