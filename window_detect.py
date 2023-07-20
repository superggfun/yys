"""
window_detect模块，提供了WindowDetect类，用于进行物体检测和点击操作。
"""
import time
import random
import sys
import os
from pathlib import Path

import torch
import pyautogui

from models.common import DetectMultiBackend
from utils.dataloaders import LoadScreenshots
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class WindowDetect:
    """
    WindowDetect类，用于进行物体检测和点击操作。
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

    def __init__(self, window_name):
        """
        初始化WindowDetect对象。

        :param window_name: 窗口名称。
        :param imgsz: 输入图像的大小。
        """
        self.window_name = window_name
        self.imgsz = WindowDetect.imgsz
        self.time_profile = (Profile(), Profile(), Profile())
        self.class_names = ""

        # 如果模型还没有被加载，则进行加载
        if WindowDetect.model is None:
            WindowDetect.model = DetectMultiBackend(WindowDetect.weights, device=WindowDetect.device, dnn=False, data=WindowDetect.data, fp16=False)
            WindowDetect.stride, WindowDetect.names, WindowDetect.pytorch_tensor = WindowDetect.model.stride, WindowDetect.model.names, WindowDetect.model.pt
            self.imgsz = check_img_size(WindowDetect.imgsz, s=WindowDetect.stride)  # Check image size

            # Warmup the model
            WindowDetect.model.warmup(imgsz=(1 if WindowDetect.pytorch_tensor or WindowDetect.model.triton else 1, 3, *self.imgsz))
        self.dataset = LoadScreenshots(self.window_name, img_size=self.imgsz, stride=WindowDetect.stride, auto=WindowDetect.pytorch_tensor)

    def preprocess_image(self, image):
        """对图像进行预处理"""
        with self.time_profile[0]:
            image = torch.from_numpy(image).to(self.model.device)
            image = image.half() if self.model.fp16 else image.float()
            image /= 255
            if len(image.shape) == 3:
                image = image[None]
        return image

    def start_detect(self):
        """
        进行物体检测，并返回检测到的物体的列表。

        :return: 一个字典列表，每个字典代表一个检测到的物体，包含类别名称，边界框和置信度。
        """

        for path, image, im0s, _ in self.dataset:
            # 延迟检测
            time.sleep(0.5)
            image = self.preprocess_image(image)
            # 推断
            with self.time_profile[1]:
                pred = self.model(image, augment=False, visualize=False)
            # 进行NMS
            with self.time_profile[2]:
                pred = non_max_suppression(pred, WindowDetect.conf_thres, WindowDetect.iou_thres, WindowDetect.classes,
                                            WindowDetect.agnostic_nms, max_det=WindowDetect.max_det)
            # 处理检测结果
            for _, det in enumerate(pred):
                if len(det):
                    _, im0, _ = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
                    # 缩放边界框
                    det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], im0.shape).round()
                    # 收集检测到的物体
                    for *xyxy, conf, cls in reversed(det):
                        if self.class_names.strip() == self.names[int(cls)].strip():
                            print(f'{self.class_names}: {conf} {xyxy}')
                            return {'class_name': self.class_names, 'bbox': xyxy, 'confidence': conf}
        return None

    def detect_and_click(self, class_name, perform_click):
        """
        进行物体检测，并在检测到给定类别的物体时执行点击操作。

        :param class_name: 要检测和点击的物体的类别名称。
        :param perform_click: 用于执行点击操作的函数。
        """
        self.class_names = class_name
        detected_object = self.start_detect()
        if detected_object is not None and detected_object['class_name'] == self.class_names:
            perform_click(detected_object['bbox'])


    def perform_click(self, bbox, generate_click_position):
        """
        执行点击操作。根据给定的生成点击位置的函数，在物体的特定区域进行点击。

        :param bbox: 物体的边界框。
        :param generate_click_position: 用于生成点击位置的函数。
        """
        # 获取点击位置
        click_x, click_y = generate_click_position(bbox)

        # 生成鼠标移动后的随机休眠时间（0.5到0.75秒之间）
        sleep_time = random.uniform(0.5, 0.75)

        # 休眠随机时间
        time.sleep(sleep_time)

        # 生成鼠标移动的随机持续时间（0.1到0.3秒之间）
        duration = random.uniform(0.1, 0.3)

        # 将鼠标移动到随机位置，持续随机时间
        pyautogui.moveTo(click_x, click_y, duration=duration)
        pyautogui.click()

    def perform_click_center(self, bbox):
        """
        对物体进行点击操作。点击位置为物体的中心点。

        :param bbox: 物体的边界框。
        """
        return self.perform_click(bbox, self.generate_click_position_center)

    def perform_click_all(self, bbox):
        """
        对物体进行点击操作。点击位置为窗口的全部可点击区域。

        :param bbox: 物体的边界框。
        """
        return self.perform_click(bbox, self.generate_click_position_all)

    def generate_click_position_center(self, bbox):
        # 计算物体中心的点击位置
        """
        执行点击操作。根据给定的边界框，在物体的中心区域进行点击。

        :param bbox: 物体的边界框。
        """
        # 根据你的需求完成点击操作的代码。这里使用的逻辑是在物体的中心区域进行点击。
        x_1 = bbox[0].cpu().item() + self.dataset.left
        y_1 = bbox[1].cpu().item() + self.dataset.top
        x_2 = bbox[2].cpu().item() + self.dataset.left
        y_2 = bbox[3].cpu().item() + self.dataset.top

        # 计算中心
        center_x = (x_1 + x_2) / 2
        center_y = (y_1 + y_2) / 2

        # 计算宽度和高度的30%
        width_10_percent = (x_2 - x_1) * 0.3
        height_10_percent = (y_2 - y_1) * 0.3

        # 在中心附近生成随机点
        random_x = random.uniform(center_x - width_10_percent, center_x + width_10_percent)
        random_y = random.uniform(center_y - height_10_percent, center_y + height_10_percent)

        return random_x, random_y

    def generate_click_position_all(self, _):
        """
        计算并返回在物体全局范围内的一个随机点击位置。

        :param _: 输入参数，实际在此方法中并未使用。
        :return: 物体全局范围内的一个随机点击位置（坐标）。
        """
        # 计算物体全局的点击位置
        # 依据随机值，确定点击的边界范围
        if random.randint(0, 1) == 0:
            left, top, right, bottom = (self.dataset.left + 0.05 * self.dataset.width,
                                        self.dataset.top + 0.25 * self.dataset.height,
                                        self.dataset.left + 0.15 * self.dataset.width,
                                        self.dataset.top + 0.85 * self.dataset.height)
        else:
            left, top, right, bottom = (self.dataset.left + 0.85 * self.dataset.width,
                                        self.dataset.top + 0.25 * self.dataset.height,
                                        self.dataset.left + 0.95 * self.dataset.width,
                                        self.dataset.top + 0.85 * self.dataset.height)

        # 在边界范围内随机选择一个点进行点击
        random_x = random.uniform(left, right)
        random_y = random.uniform(top, bottom)

        return random_x, random_y
