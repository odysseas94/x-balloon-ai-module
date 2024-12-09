import skimage.draw


class ShapeInstance:
    name: str = None
    x: float = None
    y: float = None
    cy: float = None
    ry: float = None
    cx: float = None
    rx: float = None
    r: float = None
    width: float = None
    height: float = None
    all_points_x: []
    all_points_y: []
    class_name: str = None
    class_id: int = None
    scale: int = 1
    area: float = 0

    def __init__(self, name):
        self.name = name
        self.all_points_x = []
        self.all_points_y = []

    def canonical_points_to_all_points(self, coordinates):
        index = 0
        while index < len(coordinates) - 1:
            self.all_points_x.append(coordinates[index])
            self.all_points_y.append(coordinates[index + 1])
            index += 2

    def __str__(self):
        return self.name + ", " + self.class_name + "," + str(self.all_points_x) + str(self.all_points_y)

    def calculate_polygon_area(self):
        n = len(self.all_points_y)  # of points
        area = 0.0
        for i in range(n):
            j = (i + 1) % n

            area += self.all_points_x[i] * self.all_points_y[j]
            area -= self.all_points_x[j] * self.all_points_y[i]
        area = abs(area) / 2.0
        self.area = area
        return area

    def start_scaling(self, scale):
        self.scale = scale
        if scale == 1:
            return
        if self.x:
            self.x = self.x / self.scale
        if self.y:
            self.y = self.y / self.scale
        if self.cy:
            self.cy = self.cy / self.scale
        if self.ry:
            self.ry = self.ry / self.scale
        if self.cx:
            self.cx = self.cx / self.scale
        if self.rx:
            self.rx = self.rx / self.scale
        if self.r:
            self.r = self.r / self.scale
        if self.width:
            self.width = self.width / self.scale
        if self.height:
            self.height = self.height / self.height
        if len(self.all_points_y):
            self.all_points_y = self.scale_array(self.all_points_y)
        if len(self.all_points_x):
            self.all_points_x = self.scale_array(self.all_points_x)

    def scale_array(self, array):
        result = []
        for value in array:
            result.append(int((value - 5) / self.scale))
        return result

    def get_mask_by_shape_type(self):
        shape_name = self.name
        # sk image uses all x,y inverse as y,x
        if shape_name == "rect":
            return self.rectangle(self.y, self.x, self.width,
                                  self.height)
        elif shape_name == "circle":
            return skimage.draw.circle(self.cy, self.cx, self.r)
        elif shape_name == "ellipse":
            return skimage.draw.ellipse(self.cy, self.cx, self.ry,
                                        self.rx)
        elif shape_name == "polygon" or shape_name == "multipolygon":
            return skimage.draw.polygon(self.all_points_y, self.all_points_x)

        return None

    @staticmethod
    def rectangle(r0, c0, width, height):
        print(r0, c0, width, height)
        rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]
        return skimage.draw.polygon(rr, cc)
