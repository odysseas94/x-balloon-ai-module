import xlwt
import datetime


class ExcelCreator:
    excel_lines: list = []
    workbook: None
    sheet = None
    path = None
    classification_models = {}

    def __init__(self, excel_lines, classification_models, path):
        self.excel_lines = excel_lines
        self.workbook = xlwt.Workbook()
        time = datetime.datetime.now()
        self.sheet = self.workbook.add_sheet("IBD Statistics")
        self.classification_models = classification_models
        self.create_labels()
        self.create_excel_rows()
        self.path = path
        self.workbook.save(path + "/IBD Statistics " + time.strftime("%d-%B-%Y") + ".xls")

    def create_excel_rows(self):
        rows = 1
        for key, excel_line in self.excel_lines.items():
            columns = 0
            for value in excel_line.get_canonical_row():
                self.sheet.write(rows, columns, value)
                columns += 1
            rows += 1

    def create_labels(self):
        style = xlwt.easyxf('font: bold 1, color blue')
        index = 0
        for label in self.get_labels():
            self.sheet.write(0, index, label, style)
            index += 1

    def get_labels(self):
        labels = ["Image ID", "Height", "Width", "Scale", "Training Exists", "Validation Exists", "Testing Exists"]
        index = 0
        for key, value in self.classification_models.items():
            labels.append("Training Area Class " + value.name)
            index += 1
        index = 0
        for key, value in self.classification_models.items():
            labels.append("Testing Area Class " + value.name)
            index += 1
        index = 0
        for key, value in self.classification_models.items():
            labels.append("CPA Training Area Class " + value.name)
            index += 1
        index = 0
        for key, value in self.classification_models.items():
            labels.append("CPA Testing Area Class " + value.name)
            index += 1

        return labels
