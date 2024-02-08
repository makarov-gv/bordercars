import numpy as np
from tifffile import imread, imwrite
from mmdet.apis import inference_detector


class Detector:
    def __init__(self, model_path: str, config_path: str, device="cuda:0", chunk_size=10000, patch_size=1000,
                 only_vehicles=False, filter_size=True, filter_pos=True, size_limit=(200, 2000), pos_threshold=20):
        """
        Initialize class parameters using passed arguments and create detector object for border cars detection
        :param model_path: path to PyTorch model
        :param config_path: path to python config file
        :param device: "cpu" or "cuda:x" where x stands for GPU node number
        :param chunk_size: size of orthophotoplane chunks
        :param patch_size: size of inference patches
        :param only_vehicles: flag that defines whether non-vehicle bounding boxes are to be excluded
        :param filter_size: flag that defines whether bounding boxes are to be filtered by size (width*length)
        :param filter_pos: flag that defines whether bounding boxes are to be filtered by distance to the image border
        :param size_limit: lower and upper size (width*length) limits
        :param pos_threshold: distance from the image border threshold
        """
        if not model_path.endswith(".pth"):
            raise ValueError(model_path + " is not a PyTorch model")

        if not config_path.endswith(".py"):
            raise ValueError(config_path + " is not a python config file")

        if device != "cpu":
            if device.startswith("cuda:"):
                from mmrotate.core.post_processing import bbox_nms_rotated
                from bordercars import cuda_fix

                bbox_nms_rotated.multiclass_nms_rotated.__code__ = cuda_fix.multiclass_nms_rotated.__code__
            else:
                raise NotImplementedError("Only CPU and CUDA devices are supported")

        import mmcv
        from mmrotate.models import build_detector
        from mmcv.runner import load_checkpoint

        self.__chunk_size = chunk_size
        self.__patch_size = patch_size

        self.only_vehicles = only_vehicles
        self.filter_size = filter_size
        self.filter_pos = filter_pos
        self.size_limit = size_limit
        self.pos_threshold = pos_threshold

        config = mmcv.Config.fromfile(config_path)
        config.model.pretrained = None
        self.__detector = build_detector(config.model)
        self.__detector.CLASSES = load_checkpoint(self.__detector, model_path, map_location=device)['meta']['CLASSES']
        self.__detector.cfg = config
        self.__detector.to(device)
        self.__detector.eval()

        print("Detector initialized successfully")

    def __call__(self, ortho_path: str) -> list:
        """
        Split orthophotoplane into chunks. Inference each chunk by patches on the border only, ignoring the image center
        in the process. Unite all object classes (or only vehicles) into a single class. Filter out bounding boxes by
        size and position if correlated flags were passed during initialization
        :param ortho_path: path to orthophotoplane in TIFF image format
        :return: bounding boxes list
        """
        if not (ortho_path.endswith((".tif", ".tiff"))):
            raise ValueError(ortho_path + " is not in TIFF image format")

        self.__image = imread(ortho_path)
        width, height, _ = self.__image.shape

        if width % self.__chunk_size != 0 or height % self.__chunk_size != 0:
            raise ArithmeticError("Orthophotoplane width and height are not dividable by chunk size without remainder")

        print("Inference started on", ortho_path)

        horizontal_chunks_num = width // self.__chunk_size
        vertical_chunks_num = height // self.__chunk_size

        self.__bboxes = list()

        for i in range(vertical_chunks_num):
            for j in range(horizontal_chunks_num):
                chunk_x = j * self.__chunk_size
                chunk_y = i * self.__chunk_size

                chunk = self.__image[chunk_x:chunk_x + self.__chunk_size, chunk_y:chunk_y + self.__chunk_size]

                patches_num = self.__chunk_size // self.__patch_size

                bboxes = list()

                for k in range(patches_num):
                    patch_x = k * self.__patch_size

                    bboxes.extend(self.__inference_patch(chunk[patch_x:patch_x + self.__patch_size,
                                                         0:self.__patch_size], patch_x, 0))

                    bboxes.extend(self.__inference_patch(chunk[patch_x:patch_x + self.__patch_size,
                                                         self.__chunk_size - self.__patch_size:self.__chunk_size],
                                                         patch_x, self.__chunk_size - self.__patch_size))

                for k in range(patches_num - 2):
                    patch_y = (k + 1) * self.__patch_size

                    bboxes.extend(self.__inference_patch(chunk[0:self.__patch_size,
                                                         patch_y:patch_y + self.__patch_size], 0, patch_y))

                    bboxes.extend(self.__inference_patch(chunk[self.__chunk_size - self.__patch_size:self.__chunk_size,
                                                         patch_y:patch_y + self.__patch_size],
                                                         self.__chunk_size - self.__patch_size, patch_y))

                if self.filter_size:
                    bboxes = self.__filter_size(bboxes)

                if self.filter_pos:
                    bboxes = self.__filter_pos(chunk, bboxes)

                for bbox in bboxes:
                    bbox[0] += chunk_y
                    bbox[1] += chunk_x

                self.__bboxes.extend(bboxes)

                print("Progress: ", j + i * horizontal_chunks_num + 1, "/", horizontal_chunks_num * vertical_chunks_num,
                      " chunks finished", sep="")

        print("Inference finished")

        return self.__bboxes

    @staticmethod
    def save(bboxes: list, save_path: str):
        """
        Save bounding boxes list as a text file
        :param bboxes: bounding boxes list
        :param save_path: path to save bounding boxes to with .txt extension
        """
        if not save_path.endswith(".txt"):
            raise ValueError(save_path + " is not with .txt extension")

        with open(save_path, 'w') as file:
            for bbox in bboxes:
                np.savetxt(file, bbox.reshape(1, 6), fmt="%f", newline='\n')

        print("Saved bounding boxes list as", save_path)

    @staticmethod
    def load(load_path: str) -> list:
        """
        Load bounding boxes list from a text file
        :param load_path: path to load bounding boxes from with .txt extension
        """
        if not load_path.endswith(".txt"):
            raise ValueError(load_path + " is not with .txt extension")

        bboxes = []
        with open(load_path, 'r') as file:
            for line in file:
                bboxes.append(np.array([float(x) for x in line.split()], dtype=np.float32))

        print("Loaded bounding boxes list from", load_path)

        return bboxes

    @staticmethod
    def draw(bboxes: list, ortho_path: str, save_path: str):
        """
        Draw bounding boxes from a list on orthophotoplane and save it
        :param bboxes: bounding boxes list
        :param ortho_path: path to orthophotoplane in TIFF image format
        :param save_path: path to save orthophotoplane with drawn bounding boxes to in TIFF image format
        """
        if not (ortho_path.endswith((".tif", ".tiff"))):
            raise ValueError(ortho_path + " is not in TIFF image format")

        if not (save_path.endswith((".tif", ".tiff"))):
            raise ValueError(save_path + " is not in TIFF image format")

        import cv2
        ortho = imread(ortho_path)
        height, width, _ = ortho.shape

        for bbox in bboxes:
            xc, yc, w, h, ag = bbox[:5]
            wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
            hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)

            p1 = [xc - wx - hx, yc - wy - hy]
            p2 = [xc + wx - hx, yc + wy - hy]
            p3 = [xc + wx + hx, yc + wy + hy]
            p4 = [xc - wx + hx, yc - wy + hy]

            ortho = cv2.polylines(ortho, [np.array([p1, p2, p3, p4], dtype=np.int32)], True, (255, 0, 0), 2)

        imwrite(save_path, ortho, compression="adobe_deflate")

        print("Drawn bounding boxes on ", ortho_path, ", saved as ", save_path, sep="")

    @property
    def chunk_size(self):
        return self.__chunk_size

    @property
    def patch_size(self):
        return self.__patch_size

    def __inference_patch(self, patch, patch_x, patch_y):
        bboxes = self.__unite_classes(inference_detector(self.__detector, patch))

        for bbox in bboxes:
            bbox[0] += patch_y
            bbox[1] += patch_x

        return bboxes

    def __unite_classes(self, bboxes):
        if self.only_vehicles:
            united_bboxes = list()

            for small_vehicle in bboxes[4]:
                united_bboxes.append(small_vehicle)

            for large_vehicle in bboxes[5]:
                united_bboxes.append(large_vehicle)
        else:
            united_bboxes = list()

            for i in range(len(bboxes)):
                for bbox in bboxes[i]:
                    united_bboxes.append(bbox)

        return united_bboxes

    def __filter_size(self, bboxes):
        filtered_bboxes = list()

        for bbox in bboxes:
            if self.size_limit[0] < bbox[2] * bbox[3] < self.size_limit[1]:
                filtered_bboxes.append(bbox)

        return filtered_bboxes

    def __filter_pos(self, chunk, bboxes):
        width, height, _ = chunk.shape
        filtered_bboxes = list()

        for bbox in bboxes:
            if ((bbox[0] < self.pos_threshold or bbox[0] > width - self.pos_threshold)
                    or (bbox[1] < self.pos_threshold or bbox[1] > height - self.pos_threshold)):
                filtered_bboxes.append(bbox)

        return filtered_bboxes
