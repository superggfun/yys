"""
window_detect模块，提供了WindowDetect类，用于进行物体检测和点击操作。
"""
import time
import random
import sys
import numpy as np
import os
from pathlib import Path

import torch
import pyautogui

from models.common import DetectMultiBackend
from utils.dataloaders import LoadScreenshots
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from win32api import MAKELONG, SendMessage
from win32con import WM_LBUTTONUP, WM_LBUTTONDOWN, WM_ACTIVATE, WA_ACTIVE


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

    def __init__(self, window_name, use_sct=True):
        """
        初始化WindowDetect对象。

        :param window_name: 窗口名称。
        :param use_sct: 是否使用 mss 库（不能遮挡）进行截图，默认为 True，如果为 False，将使用 Windows API 进行后台截图
        """
        self.window_name = window_name
        self.imgsz = WindowDetect.imgsz
        self.time_profile = (Profile(), Profile(), Profile())
        self.class_names = ""
        self.use_sct = use_sct

        # 如果模型还没有被加载，则进行加载
        if WindowDetect.model is None:
            WindowDetect.model = DetectMultiBackend(WindowDetect.weights, device=WindowDetect.device, dnn=False, data=WindowDetect.data, fp16=False)
            WindowDetect.stride, WindowDetect.names, WindowDetect.pytorch_tensor = WindowDetect.model.stride, WindowDetect.model.names, WindowDetect.model.pt
            self.imgsz = check_img_size(WindowDetect.imgsz, s=WindowDetect.stride)  # Check image size

            # Warmup the model
            WindowDetect.model.warmup(imgsz=(1 if WindowDetect.pytorch_tensor or WindowDetect.model.triton else 1, 3, *self.imgsz))

        # use_sct 是否使用 mss 库进行截图
        self.dataset = LoadScreenshots(self.window_name, img_size=self.imgsz, stride=WindowDetect.stride, auto=WindowDetect.pytorch_tensor, use_sct=self.use_sct)

    def preprocess_image(self, image):
        """对图像进行预处理"""
        with self.time_profile[0]:
            image = torch.from_numpy(image).to(self.model.device)
            image = image.half() if self.model.fp16 else image.float()
            image /= 255
            if len(image.shape) == 3:
                image = image[None]
        return image

    def start_detect(self, class_names, timeout=None):
        """
        进行物体检测，并返回检测到的物体。

        :param class_names: 要检测的物体的类别名称列表。
        :param timeout: 超时时间（秒）。如果设置了超时时间，将在超时时间内返回None。
        :return: 一个字典，代表一个检测到的物体，包含类别名称，边界框和置信度。
        """
        start_time = time.time() if timeout is not None else None

        for path, image, im0s, _ in self.dataset:
            # 延迟检测
            time.sleep(0.2)

            # 检查是否超时
            if start_time is not None and time.time() - start_time > timeout:
                break

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
                        if self.names[int(cls)].strip() in [name.strip() for name in class_names]:
                            print({'class_name': self.names[int(cls)], 'bbox': xyxy, 'confidence': conf})
                            return {'class_name': self.names[int(cls)], 'bbox': xyxy, 'confidence': conf}
        return None


    def detect_and_click(self, class_name, perform_click, timeout=None, no_delay=False, double_click_probability=True):
        """
        进行物体检测，并在检测到给定类别的物体时执行点击操作。

        :param class_name: 要检测和点击的物体的类别名称。
        :param perform_click: 用于执行点击操作的函数。
        :param timeout: 超时时间（秒）。如果为None，则没有超时时间。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        :return: 布尔值，如果检测到物体并执行了点击操作，则返回True，否则返回False。
        """
        detected_object = self.start_detect([class_name], timeout)
        if detected_object:
            perform_click(detected_object['bbox'], no_delay=no_delay, double_click_probability=double_click_probability)
        return detected_object is not None

    def detect_and_click_any(self, class_names, perform_click, no_delay=False, double_click_probability=True):
        """
        进行物体检测，并在检测到给定类别列表中的任意物体时执行点击操作。

        :param class_names: 要检测和点击的物体的类别名称列表。
        :param perform_click: 用于执行点击操作的函数。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        :return: 布尔值，如果检测到物体并执行了点击操作，则返回True，否则返回False。
        """
        detected_object = self.start_detect(class_names)
        if detected_object:
            perform_click(detected_object['bbox'], no_delay=no_delay, double_click_probability=double_click_probability)
        return detected_object is not None

    def detect_and_click_priority(self, class_priorities, perform_click, no_delay=False, double_click_probability=True):
        """
        对指定的多个类别进行物体检测，并在检测到优先级最高的类别的物体时执行点击操作。

        优先级通过传递的字典来确定。字典的键为类别名称，值为优先级数值。优先级数值越大，优先级越高。
        例如，传入 {"cat": 1, "dog": 2, "bird": 3}，"bird"将具有最高的优先级，其次是"dog"，最后是"cat"。

        :param class_priorities: 要检测和点击的物体的类别名称和对应的优先级，形式为 {类别名称: 优先级}。
        :param perform_click: 用于执行点击操作的函数。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        :return: 如果检测到物体并执行了点击操作，则返回对应的类别名称，否则返回None。
        """
        # 按优先级对类别进行排序，优先级高的类别排在前面
        sorted_class_priorities = sorted(class_priorities.items(), key=lambda x: x[1], reverse=True)
        
        # 逐个检测每一个类别
        for class_name, _ in sorted_class_priorities:
            detected_object = self.start_detect([class_name], timeout=1)  # 设置一个较短的超时时间
            if detected_object:
                perform_click(detected_object['bbox'], no_delay=no_delay, double_click_probability=double_click_probability)
                return class_name  # 返回识别的类别名称

        return None  # 如果没有检测到任何物体，返回None



    def perform_click(self, bbox, generate_click_position, no_delay=False, double_click_probability=True):
        """
        执行点击操作。根据给定的生成点击位置的函数，在物体的特定区域进行点击。
        根据 use_sct 参数的设置，选择前台或者后台点击。

        :param bbox: 物体的边界框。
        :param generate_click_position: 用于生成点击位置的函数。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        """
        if self.use_sct:
            self.perform_click_foreground(bbox, generate_click_position, no_delay, double_click_probability)
        else:
            self.perform_click_background(bbox, generate_click_position)


    def perform_click_background(self, bbox, generate_click_position):
        """
        执行后台点击操作。根据给定的生成点击位置的函数，在物体的特定区域进行点击。

        :param bbox: 物体的边界框。
        :param generate_click_position: 用于生成点击位置的函数。
        """
        # 获取点击位置
        click_x, click_y = generate_click_position(bbox)

        # 获取目标窗口句柄
        hwnd = self.dataset.handle

        # 生成点击持续时间（0.5到0.75秒之间）
        sleep_time = random.uniform(0.5, 0.75)

        # 休眠随机时间
        time.sleep(sleep_time)

        # 计算点击位置
        long_position = MAKELONG(int(click_x)-self.dataset.left, int(click_y)-self.dataset.top)
        print(long_position)

        # 发送点击事件
        SendMessage(hwnd, WM_ACTIVATE, WA_ACTIVE, 0)
        SendMessage(hwnd, WM_LBUTTONDOWN, 0, long_position)  # 模拟鼠标按下
        time.sleep(random.uniform(0.16, 0.25))  # 点击弹起改为随机
        SendMessage(hwnd, WM_LBUTTONUP, 0, long_position)  # 模拟鼠标弹起

    def perform_click_foreground(self, bbox, generate_click_position, no_delay=False, double_click_probability=True):
        """
        执行前台点击操作。根据给定的生成点击位置的函数，在物体的特定区域进行点击。

        :param bbox: 物体的边界框。
        :param generate_click_position: 用于生成点击位置的函数。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        """
        # 获取点击位置
        click_x, click_y = generate_click_position(bbox)

        if not no_delay:
            # 生成鼠标移动后的随机休眠时间（0.5到0.75秒之间）
            sleep_time = random.uniform(0.5, 0.75)

            # 休眠随机时间
            time.sleep(sleep_time)

        # 生成鼠标移动的随机持续时间（0.1到0.3秒之间）
        duration = random.uniform(0.1, 0.3)

        # 将鼠标移动到随机位置，持续随机时间
        pyautogui.moveTo(click_x, click_y, duration=duration)
        pyautogui.click()

        if double_click_probability and random.random() < 0.2:
            if not no_delay:
                # 生成双击之间的随机休眠时间（0.1到0.15秒之间）
                double_click_sleep_time = random.uniform(0.1, 0.15)

                # 休眠随机时间
                time.sleep(double_click_sleep_time)

            pyautogui.click()



    def perform_click_center(self, bbox, no_delay=False, double_click_probability=True):
        """
        对物体进行点击操作。点击位置为物体的中心点。

        :param bbox: 物体的边界框。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        """
        return self.perform_click(bbox, self.generate_click_position_center, no_delay=no_delay, double_click_probability=double_click_probability)

    def perform_click_all(self, bbox, no_delay=False, double_click_probability=True):
        """
        对物体进行点击操作。点击位置为窗口的全部可点击区域。

        :param bbox: 物体的边界框。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        """
        return self.perform_click(bbox, self.generate_click_position_all, no_delay=no_delay, double_click_probability=double_click_probability)

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
                                        self.dataset.top + 0.65 * self.dataset.height)
        else:
            left, top, right, bottom = (self.dataset.left + 0.85 * self.dataset.width,
                                        self.dataset.top + 0.25 * self.dataset.height,
                                        self.dataset.left + 0.95 * self.dataset.width,
                                        self.dataset.top + 0.85 * self.dataset.height)

        # 在边界范围内随机选择一个点进行点击
        random_x = random.uniform(left, right)
        random_y = random.uniform(top, bottom)

        return random_x, random_y
    
    def swipe_screen(self, is_left_to_right=False):
        """
        在当前窗口内随机执行一个滑动操作。

        :param is_left_to_right: 滑动的方向。True表示从左向右滑动，False表示从右向左滑动。
        :return: None
        """
        # 为了保证滑动操作在窗口内进行，我们设置滑动起始点和终止点的范围，这里假设滑动的区域距离窗口边缘为窗口的四分之一
        left_boundary = self.dataset.left + self.dataset.width * 1 / 4
        right_boundary = self.dataset.left + self.dataset.width * 3 / 4
        top_boundary = self.dataset.top + self.dataset.height * 1 / 4
        bottom_boundary = self.dataset.top + self.dataset.height * 3 / 4

        # 在设置的范围内随机选择滑动的起始点和终止点的y坐标（保持在同一水平线上）
        y_start = y_end = random.uniform(top_boundary, bottom_boundary)

        if is_left_to_right:
            # 滑动的起始点的x坐标在左半部分随机选择
            x_start = random.uniform(left_boundary, (left_boundary + right_boundary) / 2)
            # 滑动的终止点的x坐标在滑动起始点的右边随机选择一个距离，距离在窗口宽度的三分之一左右
            x_end = x_start + random.uniform(self.dataset.width * 1 / 6, self.dataset.width * 1 / 3)
            x_end = min(x_end, right_boundary)
        else:
            # 滑动的起始点的x坐标在右半部分随机选择
            x_start = random.uniform((left_boundary + right_boundary) / 2, right_boundary)
            # 滑动的终止点的x坐标在滑动起始点的左边随机选择一个距离，距离在窗口宽度的三分之一左右
            x_end = x_start - random.uniform(self.dataset.width * 1 / 6, self.dataset.width * 1 / 3)
            x_end = max(x_end, left_boundary)

        # 先移动到起始位置
        pyautogui.moveTo(x_start, y_start)

        # 执行滑动操作，你可以根据需要调整滑动的速度（第三个参数）
        swipe_duration = random.uniform(0.5, 1)  # 随机滑动时间
        pyautogui.dragTo(x_end, y_end, button='left', duration=swipe_duration)
