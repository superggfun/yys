"""
@superggfun
创建日期: 2023-07-18
版本: 1.0.0
描述: 基于Yolov5的阴阳师识别点击程序。
"""


from sys import executable
import os
import ctypes
import sys
from pathlib import Path
import time
import random
import torch
import pyautogui

from models.common import DetectMultiBackend
from utils.dataloaders import LoadScreenshots
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@smart_inference_mode()
def run(
        weights=ROOT / 'train/weights/best.pt',
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml 路径
        imgsz=(640, 640),  # 推理大小 (高度, 宽度)
        conf_thres=0.7,  # 置信度阈值
        iou_thres=0.45,  # NMS IoU阈值
        max_det=1000,  # 每张图像的最大检测数
        device='',  # CUDA设备，例如 0 或 0,1,2,3 或 cpu
        classes=None,  # 按类别过滤：--class 0 或 --class 0 2 3
        agnostic_nms=False,  # 类别不敏感的NMS
):
    """
    以给定参数运行模型。

    参数:
    weights: 模型权重路径.
    data: 数据集路径.
    imgsz: 推理的大小.
    conf_thres: 置信度阈值.
    iou_thres: NMS IoU阈值.
    max_det: 每张图像的最大检测数.
    device: 运行模型的设备.
    classes: 需要过滤的类别.
    agnostic_nms: 是否使用类别不敏感的NMS.

    返回:
    无
    """

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pytorch_tensor = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    batch_size = 1  # batch_size

    # 设定你想要按顺序检测的类别名称
    class_names = ['tili']
    # 初始化下一个要检测的类别的索引
    next_class_index = 0

    # 加载截图(窗口名字)
    dataset = LoadScreenshots("阴阳师-网易游戏", img_size=imgsz, stride=stride, auto=pytorch_tensor)

    while True:  # 主循环，持续进行截屏和检测
        # 预热模型
        model.warmup(imgsz=(1 if pytorch_tensor or model.triton else batch_size, 3, *imgsz))
        _, _, time_profile = 0, [], (Profile(), Profile(), Profile())
        for path, image, im0s, _ in dataset:
            # 延迟检测
            time.sleep(0.5)
            with time_profile[0]:
                # 图像预处理...
                image = torch.from_numpy(image).to(model.device)  # pylint: disable=no-member
                image = image.half() if model.fp16 else image.float()  # pylint: disable=no-member
                image /= 255
                if len(image.shape) == 3:
                    image = image[None]  # 扩展batch dim
            # 进行推理
            with time_profile[1]:
                pred = model(image, augment=False, visualize=False)

            # 进行NMS
            with time_profile[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # 处理预测结果
            for _, det in enumerate(pred):
                if len(det):
                    _, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    # 缩放检测框大小
                    det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], im0.shape).round()

                    # 检查当前待检测的类别是否被检测到
                    for *xyxy, conf, cls in reversed(det):
                        if names[int(cls)].strip() == class_names[next_class_index].strip():
                            # 执行你的点击操作或其他操作...
                            # 点击并跳出循环
                            x_1 = xyxy[0].cpu().item() + dataset.left
                            y_1 = xyxy[1].cpu().item() + dataset.top
                            x_2 = xyxy[2].cpu().item() + dataset.left
                            y_2 = xyxy[3].cpu().item() + dataset.top

                            # 计算中心点
                            center_x = (x_1 + x_2) / 2
                            center_y = (y_1 + y_2) / 2

                            # 计算10%的宽度和高度
                            width_10_percent = (x_2 - x_1) * 0.1
                            height_10_percent = (y_2 - y_1) * 0.1

                            # 生成靠近中心的随机点
                            random_x = random.uniform(center_x - width_10_percent, center_x + width_10_percent)
                            random_y = random.uniform(center_y - height_10_percent, center_y + height_10_percent)

                            print(f'{names[int(cls)]}: {conf} {random_x, random_y}')

                            # 生成一个每次鼠标移动后的随机休眠时间（在0.5到1.5秒之间）
                            sleep_time = random.uniform(0.5, 1.5)

                            # 休眠随机时间
                            time.sleep(sleep_time)
                            # 生成一个鼠标移动的随机持续时间（在0.1到0.3秒之间）
                            duration = random.uniform(0.1, 0.3)

                            # 将鼠标移动到随机位置并持续随机时间
                            pyautogui.moveTo(random_x, random_y, duration=duration)
                            pyautogui.click()
                            pyautogui.click()

                            # 更新到下一个类别
                            next_class_index += 1
                            # 如果所有类别都已经检测完，可以在这里结束程序，或者重置next_class_index
                            if next_class_index == len(class_names):
                                next_class_index = 0
                            break  # 跳出内循环
                    else:
                        continue  # 如果当前类别没有被检测到，回到循环开始，获取新的截图并再次检测
                    break  # 如果当前类别被检测到，跳出循环并检测下一个类别
                break  # 如果当前类别被检测到，跳出循环并检测下一个类别

def is_admin():
    """
    Check if the user is an admin on a Windows system.

    Returns:
    bool: True if the user is an admin, False otherwise.
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except (OSError, AttributeError) as err:
        print(err)
    return False


def run_as_admin():
    """
    以管理员权限重新运行程序
    """
    ctypes.windll.shell32.ShellExecuteW(None, "runas", executable, __file__, None, 1)


if __name__ == '__main__':
    if is_admin():
        run()
    else:
        run_as_admin()
