from app.src.models.DatabaseModel import DatabaseModel


class WeightFile(DatabaseModel):
    id = 0
    name = ""
    success_ratio = ""
    error_ratio = ""
    configuration = ""
    val_loss = 0,
    val_rpn_class_loss = 0,
    val_rpn_bbox_loss = 0,
    val_mrcnn_class_loss = 0,
    val_mrcnn_bbox_loss = 0,
    val_mrcnn_mask_loss = 0,
    rpn_class_loss = 0,
    rpn_bbox_loss = 0,
    mrcnn_class_loss = 0,
    mrcnn_bbox_loss = 0,
    mrcnn_mask_loss = 0
    path = ""
    date_created = ""
    date_updated = ""

    def __init__(self, id, path, name, success_ratio, configuration, val_loss, val_rpn_class_loss,
                 val_rpn_bbox_loss, val_mrcnn_class_loss, val_mrcnn_bbox_loss, val_mrcnn_mask_loss, rpn_class_loss,
                 rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, error_ratio, date_created,
                 date_updated, *args, **kwargs):
        super().__init__("weight_file")
        self.id = id
        self.name = name
        self.path = path
        self.configuration = configuration
        self.val_loss = val_loss
        self.val_rpn_class_loss = val_rpn_class_loss
        self.val_rpn_bbox_loss = val_rpn_bbox_loss
        self.val_mrcnn_class_loss = val_mrcnn_class_loss
        self.val_mrcnn_bbox_loss = val_mrcnn_bbox_loss
        self.val_mrcnn_mask_loss = val_mrcnn_mask_loss
        self.rpn_class_loss = rpn_class_loss
        self.rpn_bbox_loss = rpn_bbox_loss
        self.mrcnn_class_loss = mrcnn_class_loss
        self.mrcnn_bbox_loss = mrcnn_bbox_loss
        self.mrcnn_mask_loss = mrcnn_bbox_loss
        self.success_ratio = success_ratio
        self.error_ratio = error_ratio
        self.date_created = date_created
        self.date_updated = date_updated

    def __str__(self) -> str:
        return str(self.id) + " : " + self.name + " : " + self.path + " : " + str(self.success_ratio)

    def get_attributes(self):
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "configuration": self.configuration,
            "val_loss": self.val_loss,
            "val_rpn_class_loss": self.val_rpn_class_loss,
            "val_rpn_bbox_loss": self.val_rpn_bbox_loss,
            "val_mrcnn_class_loss": self.val_mrcnn_class_loss,
            "val_mrcnn_bbox_loss": self.val_mrcnn_bbox_loss,
            "val_mrcnn_mask_loss": self.val_mrcnn_mask_loss,
            "rpn_class_loss": self.rpn_class_loss,
            "rpn_bbox_loss": self.rpn_bbox_loss,
            "mrcnn_class_loss": self.mrcnn_class_loss,
            "mrcnn_bbox_loss": self.mrcnn_bbox_loss,
            "mrcnn_mask_loss": self.mrcnn_bbox_loss,
            "success_ratio": self.success_ratio,
            "error_ratio": self.error_ratio,
        }
