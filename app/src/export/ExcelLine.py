from app.src.export.ContainerImage import ContainerImage
from app.src.export.SetInstancesExport import SetInstancesExport
from app.src.models.ImageModel import ImageModel


class ExcelLine:
    image_model: ImageModel
    image_id: int
    training_exists: bool
    validation_exists: bool
    testing_exists: bool
    areas = {}
    testing_areas = {}
    training_area_list = []
    testing_area_list = []
    cpa_training = []
    cpa_testing = []
    classification_models = {}
    container_image: ContainerImage = None
    testing_image_model: ImageModel = None

    set_instances_export: SetInstancesExport

    def __init__(self, image_model: ImageModel, set_instances_export: SetInstancesExport, classification_models,
                 container_image: ContainerImage):
        self.image_model = image_model
        self.container_image = container_image
        self.image_id = image_model.id
        self.training_area_list = []
        self.testing_area_list = []
        self.areas = {}
        self.cpa_testing = []
        self.cpa_training = []
        self.testing_areas = {}
        self.classification_models = classification_models
        self.set_instances_export = set_instances_export
        self.check_data_set()
        self.init_area()

    def init_area(self):
        for key, value in self.classification_models.items():
            self.areas[key] = 0
            self.testing_areas[key] = 0
            self.training_area_list.append(0)
            self.testing_area_list.append(0)
            self.cpa_testing.append(0)
            self.cpa_training.append(0)

    def calculate_areas_by_class_training(self):
        for shape in self.image_model.shapes:
            self.areas[shape.class_id] += shape.calculate_polygon_area() * self.container_image.scale
        index = 0
        for key, value in self.classification_models.items():
            self.training_area_list[index] = (int(self.areas[key]))
            index += 1
        print(self.get_canonical_row())

    def calculate_areas_by_class_testing(self, testing_image_model):
        self.testing_image_model = testing_image_model
        for shape in self.testing_image_model.shapes:
            self.testing_areas[shape.class_id] += shape.calculate_polygon_area() * self.container_image.scale
        index = 0
        for key, value in self.classification_models.items():
            self.testing_area_list[index] = (int(self.testing_areas[key]))
            index += 1
        print(self.get_canonical_row())

    def check_data_set(self):
        self.training_exists = self.check_image_in_array(self.set_instances_export.training_instance.image_models)
        self.validation_exists = self.check_image_in_array(self.set_instances_export.validation_instance.image_models)
        self.testing_exists = self.check_image_in_array(self.set_instances_export.testing_instance.image_models)

    def check_image_in_array(self, array):
        for image in array:
            if image == self.image_model:
                return True
        return False

    def set_cpa_training(self, cpa_training):
        index = 0
        for value in cpa_training:
            self.cpa_training[index] = value * self.container_image.scale
            index += 1

    def set_cpa_testing(self, cpa_testing):
        index = 0
        for value in cpa_testing:
            self.cpa_testing[index] = value * self.container_image.scale
            index += 1

    def get_canonical_row(self):
        result = [self.image_id, self.container_image.original_height, self.container_image.original_width,
                  int(self.container_image.scale), self.training_exists, self.validation_exists, self.testing_exists,
                  self.training_area_list, self.testing_area_list, self.cpa_training, self.cpa_testing]
        return flatten_list(result)


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list
