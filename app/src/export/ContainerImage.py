from app.src.library.KMeansCluster import KMeansCluster
from app.src.models.ImageModel import ImageModel
import cv2


class ContainerImage:
    image_model: ImageModel = None
    scale = 1
    export_path = ""

    actual_image = None
    k_means_cluster: KMeansCluster = None
    path = None
    original_height = 0
    original_width = 0
    resized_image_path = None
    resized_image_height = 0
    resized_image_width = 0

    def __init__(self, image_model: ImageModel, path, export_path):
        self.image_model = image_model
        self.path = path
        self.export_path = export_path
        self.resize_image_save()
        self.k_means_cluster = KMeansCluster(self.actual_image,
                                             self.export_path + "/" + "cpa-k-means.jpg")
        self.k_means_cluster.detect()

        self.image_model.scale_all_shapes(self.scale)

    def resize_image_save(self):
        image_original = cv2.imread(self.path + self.image_model.canonical_name)
        original_width = image_original.shape[1]
        height = image_original.shape[0]
        self.original_height = height
        self.original_width = original_width
        width = 1080
        self.resized_image_width = width
        self.scale = original_width / width
        self.resized_image_height = int(height / self.scale)
        self.actual_image = cv2.resize(image_original, (width, self.resized_image_height), interpolation=cv2.INTER_AREA)
        self.resized_image_path = self.export_path + "/" + "original" + self.image_model.extension
        cv2.imwrite(self.resized_image_path, self.actual_image)
