from app.src.export.ContainerImage import ContainerImage
from app.src.export.ExcelCreator import ExcelCreator
from app.src.export.ExcelLine import ExcelLine
from app.src.export.ImagePainter import ImagePainter
from app.src.export.SetInstancesExport import SetInstancesExport
from app.src.instances.LoadFromServer import LoadFromServer
import os
from pathlib import Path
from PIL import ImageColor

GLOBAL_PATH = os.path.abspath(__file__ + "../../../../")


class ExportHandler:
    testing_instance = None
    training_instance = None
    validation_instance = None
    load_from_server = None
    load_from_server_testing = None
    export_path = ""
    set_instance_export: SetInstancesExport
    classification_models = {}
    excel_rows = {}

    def __init__(self, load_from_server_testing=None):
        self.load_from_server_testing = load_from_server_testing
        if not load_from_server_testing:
            self.load_from_server_testing = LoadFromServer("testing")
            self.testing_instance = self.load_from_server_testing.testingInstance
        else:
            self.testing_instance = self.load_from_server_testing.testingInstance
        self.load_from_server = LoadFromServer()
        self.training_instance = self.load_from_server.trainingInstance
        self.validation_instance = self.load_from_server.validationInstance
        self.classification_models = self.load_from_server.classifications
        if not self.load_from_server_testing.testingModel:
            print("Testing not found. Try again when you change testing set")
        else:
            print(self.training_instance, self.validation_instance, self.testing_instance)
            self.export_path += GLOBAL_PATH + "/exports/testing-" + str(self.load_from_server_testing.testingModel.id)
            self.set_instance_export = SetInstancesExport(self.load_from_server, self.load_from_server_testing)
            self.create_folders()
            self.init_instances_training()

    def create_folders(self):
        Path(GLOBAL_PATH + "/exports").mkdir(parents=True, exist_ok=True)
        Path(self.export_path).mkdir(parents=True, exist_ok=True)

        for image in self.testing_instance.image_models:
            save_path = self.export_path + "/" + str(image.id)
            image.saved_path = save_path
            Path(save_path).mkdir(parents=True, exist_ok=True)

    def init_instances_training(self):
        for image in self.testing_instance.image_models:

            fetched_image_model = self.find_image_from_training(image)

            if fetched_image_model:
                fetched_image_model.saved_path = image.saved_path
                print("fetched_image_model", len(fetched_image_model.shapes))
            container_image = ContainerImage(fetched_image_model, self.testing_instance.current_path, image.saved_path)
            excel_line = ExcelLine(fetched_image_model, self.set_instance_export, self.classification_models,
                                   container_image)
            image_painter = ImagePainter(container_image, self.classification_models, "training")
            excel_line.set_cpa_training(image_painter.cpa_area_by_classification)
            excel_line.calculate_areas_by_class_training()
            self.excel_rows[image.id] = excel_line
        self.save_excel_export()

    def save_excel_export(self):
        ExcelCreator(self.excel_rows, self.classification_models, self.export_path)

    def init_instances_testing_image(self, image):

        container_image = ContainerImage(image, self.testing_instance.current_path, image.saved_path)
        image_painter = ImagePainter(container_image, self.classification_models, "testing")
        excel_line = self.excel_rows[image.id]
        excel_line.set_cpa_testing(image_painter.cpa_area_by_classification)
        excel_line.calculate_areas_by_class_testing(image)

    def find_image_from_training(self, testing_image):

        for image in self.training_instance.image_models:
            if image == testing_image:
                return image
        for image in self.validation_instance.image_models:
            if image == testing_image:
                print("validation image found")
                return image
        return testing_image

# exportHandler = ExportHandler()
