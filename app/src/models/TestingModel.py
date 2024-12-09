from app.src.models.DatabaseModel import DatabaseModel


class TestingModel(DatabaseModel):
    active = True
    dataset_id = 1
    date_created = 0
    date_updated = 0
    description = ""
    id = 0
    name = 0
    status_id = 0

    def __init__(self, id, name, description, active, dataset_id, date_created, date_updated, *args,
                 **kwargs):
        super().__init__("testing")
        self.id = id
        self.name = name
        self.description = description
        self.active = active
        self.dataset_id = dataset_id
        self.date_created = date_created
        self.date_updated = date_updated
