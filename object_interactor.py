"""
ObjectInteractor类，用于进行物体的交互操作。
"""
import time
import random
import threading
import pyautogui
from typing import Literal

from win32api import MAKELONG, SendMessage
from win32con import WM_LBUTTONUP, WM_LBUTTONDOWN, WM_ACTIVATE, WA_ACTIVE
from ppadb.client import Client as AdbClient

from object_detector import ObjectDetector


class ObjectInteractor:
    """
    ObjectInteractor类，用于进行物体的交互操作。
    """
    # 类级别的信号量，所有实例共享
    click_semaphore = threading.Semaphore(1)

    def __init__(self, window_name: str, mode: Literal['mss', 'win_api', 'adb'] = 'mss', stop_callback=None):
        """
        初始化ObjectInteractor对象。

        :param window_name: 窗口名称。
        :param mode: 截图模式，'mss'表示使用 mss 库（不能遮挡）进行截图，'win_api'表示使用 Windows API 进行后台截图，'adb'表示使用ADB进行截图。默认为'mss'
        """
        self.window_name = window_name
        self.detector = ObjectDetector(window_name, mode)
        self.dataset = self.detector.dataset
        self.stopped = False  # 添加 'stopped' 属性


    def pause(self):
        """暂停执行"""
        self.detector.pause()

    def resume(self):
        """恢复执行"""
        self.detector.resume()

    def stop(self):
        """停止执行"""
        self.running = False
        self.detector.stop()
        

    def detect_and_click(self, class_name, perform_click, timeout=None, no_delay=False, double_click_probability=True, stop_if_no_detect=False):
        """
        进行物体检测，并在检测到给定类别的物体时执行点击操作。

        :param class_name: 要检测和点击的物体的类别名称。
        :param perform_click: 用于执行点击操作的函数。
        :param timeout: 超时时间（秒）。如果为None，则没有超时时间。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        :param stop_if_no_detect: 如果设置为True，在没有检测到对象时立即返回None。
        :return: 布尔值，如果检测到物体并执行了点击操作，则返回True，否则返回False。
        """
        while not self.stopped:  # 检查停止标志
            detected_object = self.detector.start_detect([class_name], timeout, stop_if_no_detect)
            if detected_object:
                perform_click(detected_object['bbox'], no_delay, double_click_probability)
            return detected_object is not None
        return False


    def detect_and_click_any(self, class_names, perform_click, no_delay=False, double_click_probability=True, stop_if_no_detect=False):
        """
        进行物体检测，并在检测到给定类别列表中的任意物体时执行点击操作。

        :param class_names: 要检测和点击的物体的类别名称列表。
        :param perform_click: 用于执行点击操作的函数。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        :param stop_if_no_detect: 如果设置为True，在没有检测到对象时立即返回None。
        :return: 布尔值，如果检测到物体并执行了点击操作，则返回True，否则返回False。
        """
        detected_object = self.detector.start_detect(class_names, None, stop_if_no_detect)
        if detected_object:
            perform_click(detected_object['bbox'], no_delay, double_click_probability, stop_if_no_detect)
        return detected_object is not None

    def detect_and_click_priority(self, class_priorities, perform_click, no_delay=False, double_click_probability=True, stop_if_no_detect=False):
        """
        对指定的多个类别进行物体检测，并在检测到优先级最高的类别的物体时执行点击操作。

        优先级通过传递的字典来确定。字典的键为类别名称，值为优先级数值。优先级数值越大，优先级越高。
        例如，传入 {"cat": 1, "dog": 2, "bird": 3}，"bird"将具有最高的优先级，其次是"dog"，最后是"cat"。

        :param class_priorities: 要检测和点击的物体的类别名称和对应的优先级，形式为 {类别名称: 优先级}。
        :param perform_click: 用于执行点击操作的函数。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        :param stop_if_no_detect: 如果设置为True，在没有检测到对象时立即返回None。
        :return: 如果检测到物体并执行了点击操作，则返回对应的类别名称，否则返回None。
     WW   """
        # 按优先级对类别进行排序，优先级高的类别排在前面
        sorted_class_priorities = sorted(class_priorities.items(), key=lambda x: x[1], reverse=True)

        # 逐个检测每一个类别
        for class_name, _ in sorted_class_priorities:
            detected_object = self.detector.start_detect([class_name], stop_if_no_detect=stop_if_no_detect, timeout=1)  # 设置一个较短的超时时间
            if detected_object:
                perform_click(detected_object['bbox'], no_delay, double_click_probability)
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
        if self.dataset.mode == "win_api":
            self._perform_click_background(bbox, generate_click_position)
        else:
            click_x, click_y, double_click = self._click_common(bbox, generate_click_position, no_delay, double_click_probability)

            if self.dataset.mode == "mss":
                with ObjectInteractor.click_semaphore:
                    self._perform_click_foreground(click_x, click_y, double_click)
            elif self.dataset.mode == "adb":
                self._adb_perform_click(click_x, click_y, double_click)

    
    def _click_common(self, bbox, generate_click_position, no_delay=False, double_click_probability=True):
        """
        处理点击操作的公共部分。

        :param bbox: 物体的边界框。
        :param generate_click_position: 用于生成点击位置的函数。
        :param no_delay: 是否在点击后立即返回，不进行延迟。默认为False。
        :param double_click_probability: 是否有可能进行双击操作。默认为True。
        :return: 点击的坐标，双击的可能性
        """
        # 获取点击位置
        click_x, click_y = generate_click_position(bbox)

        if not no_delay:
            # 生成鼠标移动后的随机休眠时间（0.5到0.75秒之间）
            sleep_time = random.uniform(0.5, 0.75)

            # 休眠随机时间
            time.sleep(sleep_time)

        double_click = double_click_probability and random.random() < 0.2
        if double_click:
            if not no_delay:
                # 生成双击之间的随机休眠时间（0.1到0.15秒之间）
                double_click_sleep_time = random.uniform(0.1, 0.15)

                # 休眠随机时间
                time.sleep(double_click_sleep_time)
                
        return click_x, click_y, double_click




    def _perform_click_background(self, bbox, generate_click_position):
        """
        执行后台点击操作。根据给定的生成点击位置的函数，在物体的特定区域进行点击。

        :param bbox: 物体的边界框。
        :param generate_click_position: 用于生成点击位置的函数。
        """
        # 获取点击位置
        click_x, click_y = generate_click_position(bbox)

        # 获取目标窗口句柄
        hwnd = self.detector.dataset.handle

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

    def _perform_click_foreground(self, click_x, click_y, double_click):
        """
        执行前台点击操作。

        :param click_x: 点击的x坐标
        :param click_y: 点击的y坐标
        :param double_click: 是否进行双击操作。
        """
        # 生成鼠标移动的随机持续时间（0.1到0.3秒之间）
        duration = random.uniform(0.1, 0.3)

        # 将鼠标移动到随机位置，持续随机时间
        pyautogui.moveTo(click_x, click_y, duration=duration)
        pyautogui.click()

        if double_click:
            # 执行第二次点击操作
            pyautogui.click()

    def _adb_perform_click(self, click_x, click_y, double_click):
        """
        使用ADB执行点击操作。

        :param click_x: 点击的x坐标
        :param click_y: 点击的y坐标
        :param double_click: 是否进行双击操作。
        """
        # 使用adb shell命令执行点击操作
        device = AdbClient(host="127.0.0.1", port=5037).device(self.window_name)
        device.shell(f'input tap {click_x} {click_y}')

        if double_click:
            # 执行第二次点击操作
            device.shell(f'input tap {click_x} {click_y}')

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
