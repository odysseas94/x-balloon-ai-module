class DatabaseModel:
    entity_name = ""

    def __init__(self, entity_name):
        self.entity_name = entity_name

    def get_attributes(self):
        return {}


