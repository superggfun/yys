import sys
import os
from pathlib import Path
import time
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadScreenshots
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class ObjectDetector:
    """
    ObjectDetector类，用于进行物体检测。
    """
    model = None
    stride = None
    names = None
    pytorch_tensor = None
    device = select_device('')
    weights = ROOT / 'train/weights/best.pt'
    data = 'data/coco128.yaml'
    imgsz = (640, 640)

    conf_thres = 0.7
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000  # 最大检测数量

    def __init__(self, window_name, use_sct=True):
        """
        初始化ObjectDetector对象。

        :param window_name: 窗口名称。
        :param use_sct: 是否使用 mss 库（不能遮挡）进行截图，默认为 True，如果为 False，将使用 Windows API 进行后台截图
        """
        self.window_name = window_name
        self.imgsz = ObjectDetector.imgsz
        self.time_profile = (Profile(), Profile(), Profile())
        self.class_names = ""
        self.use_sct = use_sct
        # use_sct 是否使用 mss 库进行截图
        self.dataset = LoadScreenshots(self.window_name, img_size=self.imgsz, stride=ObjectDetector.stride, auto=ObjectDetector.pytorch_tensor, use_sct=self.use_sct)

        # 如果模型还没有被加载，则进行加载
        if ObjectDetector.model is None:
            ObjectDetector.model = DetectMultiBackend(ObjectDetector.weights, device=ObjectDetector.device, dnn=False, data=ObjectDetector.data, fp16=False)
            ObjectDetector.stride, ObjectDetector.names, ObjectDetector.pytorch_tensor = ObjectDetector.model.stride, ObjectDetector.model.names, ObjectDetector.model.pt
            self.imgsz = check_img_size(ObjectDetector.imgsz, s=ObjectDetector.stride)  # Check image size

            # Warmup the model
            ObjectDetector.model.warmup(imgsz=(1 if ObjectDetector.pytorch_tensor or ObjectDetector.model.triton else 1, 3, *self.imgsz))

    def _preprocess_image(self, image):
        """对图像进行预处理"""
        with self.time_profile[0]:
            image = torch.from_numpy(image).to(self.model.device)
            image = image.half() if self.model.fp16 else image.float()
            image /= 255
            if len(image.shape) == 3:
                image = image[None]
        return image

    def start_detect(self, class_names, timeout=None, stop_if_no_detect=False):
        """
        进行物体检测，并返回检测到的物体。

        :param class_names: 要检测的物体的类别名称列表。
        :param timeout: 超时时间（秒）。如果设置了超时时间，将在超时时间内返回None。
        :param stop_if_no_detect: 如果设置为True，在没有检测到对象时立即返回None。
        :return: 一个字典，代表一个检测到的物体，包含类别名称，边界框和置信度。
        """
        
        start_time = time.time() if timeout is not None else None
        # 预处理类名列表，去除空白并唯一化
        class_names_set = set(name.strip() for name in class_names)

        for path, image, im0s, _ in self.dataset:
            # 延迟检测
            time.sleep(0.2)
            # 检查是否超时
            if start_time is not None and time.time() - start_time > timeout:
                break

            image = self._preprocess_image(image)
            # 推断
            with self.time_profile[1]:
                pred = self.model(image, augment=False, visualize=False)
            # 进行NMS
            with self.time_profile[2]:
                pred = non_max_suppression(pred, ObjectDetector.conf_thres, ObjectDetector.iou_thres, ObjectDetector.classes,
                                        ObjectDetector.agnostic_nms, max_det=ObjectDetector.max_det)
            # 处理检测结果
            if len(pred):
                for _, det in enumerate(pred):
                    if len(det):
                        _, im0, _ = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
                        # 缩放边界框
                        det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], im0.shape).round()
                        # 收集检测到的物体
                        for *xyxy, conf, cls in reversed(det):
                            if self.names[int(cls)].strip() in class_names_set:
                                print({'class_name': self.names[int(cls)], 'bbox': xyxy, 'confidence': conf})
                                return {'class_name': self.names[int(cls)], 'bbox': xyxy, 'confidence': conf}
                    elif stop_if_no_detect:
                        return None
        return None
