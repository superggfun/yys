"""
Mode模块，提供了Mode类，它是WindowDetect类的子类，实现了特定的点击操作。
"""

import time
import random
from object_interactor import ObjectInteractor
from typing import Literal
from functools import wraps


def time_and_game_limit_decorator(func):
    @wraps(func)
    def wrapped_function(self, *args, **kwargs):
        start_time = time.time()
        game_count = 0

        while True:  # 添加一个无限循环
            # 检查时间和游戏次数限制
            current_time = time.time()
            if self.time_limit is not None and current_time - start_time > self.time_limit * 60:
                print("Time limit reached, stopping the game.")
                self.stop()
                break
            if self.game_limit is not None and game_count >= self.game_limit:
                print("Game limit reached, stopping the game.")
                self.stop()
                break
            if not self.running:
                break
            
            try:
                func(self, *args, **kwargs)
                game_count += 1
            except Exception as e:
                print(f"An error occurred: {e}")
                self.stop()
                break
    return wrapped_function


class Mode(ObjectInteractor):
    """
    Mode类，ObjectInteractor的子类，实现了特定的点击操作。
    """
    def __init__(self, window_name: str, mode: Literal['mss', 'win_api', 'adb'] = 'mss', game_limit=None, time_limit=None):
        super().__init__(window_name, mode)
        self.running = True  # 状态标志，表示模式是否正在运行
        self.game_limit = game_limit  # 游戏次数限制
        self.time_limit = time_limit  # 游戏时间限制

    @time_and_game_limit_decorator
    def yuhun(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        self.detect_and_click("win", self.perform_click_center)
        self.detect_and_click("hun", self.perform_click_all)
        self.detect_and_click("tiaozhan_on", self.perform_click_center)

    @time_and_game_limit_decorator
    def yuhun2(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        self.detect_and_click("win", self.perform_click_center)
        self.detect_and_click("hun", self.perform_click_all)

    @time_and_game_limit_decorator
    def tupo(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        self.detect_and_click("tupo_new", self.perform_click_center)
        self.detect_and_click("jingong", self.perform_click_center)
        self.detect_and_click("zhunbei", self.perform_click_center)
        self.detect_and_click_any(["tupo_fail", "hun"], self.perform_click_all)
        time.sleep(1)
        self.detect_and_click("hun", self.perform_click_all, 1)

    @time_and_game_limit_decorator
    def chi(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        self.detect_and_click("tiaozhan_chi", self.perform_click_center)
        #self.detect_and_click("zhunbei", self.perform_click_center)
        self.detect_and_click("hun", self.perform_click_all)

    @time_and_game_limit_decorator
    def qiling(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        self.detect_and_click("tancha_qiling", self.perform_click_center)
        self.detect_and_click("zhunbei", self.perform_click_center, 4)
        self.detect_and_click("hun", self.perform_click_all)

    @time_and_game_limit_decorator
    def fuben(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        self.detect_and_click("instances_28", self.perform_click_center, 3)
        print(1)
        self.detect_and_click("tansuo_botton", self.perform_click_center)
        print(2)
        while True:  # 内部循环，处理 "attack_head" 和 "attack"
            print(3)
            #self.swipe_screen()
            name = self.detect_and_click_priority({"attack_head": 2, "attack": 1}, self.perform_click_center, no_delay=True)
            self.detect_and_click("attack", self.perform_click_center, 1)
            if name is None:  # 添加了错误处理，如果detect_and_click_priority函数返回None（可能是超时或者出错），那么打印一个消息并跳出内部循环
                print("No Detection!")
                self.swipe_screen()
                continue
            #if self.detect_and_click("zhunbei", self.perform_click_center, 5):
            #    print(7)
            #    continue
            if not self.detect_and_click("win", self.perform_click_all, 25):
                continue
            self.detect_and_click("hun", self.perform_click_all, 2)
            print(4)
            if name == "attack_head":
                print(5)
                time.sleep(2)
                for _ in range(4):
                    self.detect_and_click("gift", self.perform_click_center, 1)
                    self.detect_and_click("gain_gift", self.perform_click_all, 1)
                break  # 如果检测到 "attack_head"，则跳出内部循环
            time.sleep(random.uniform(0, 1))  # 在循环结束时暂停，以避免过快的循环


    @time_and_game_limit_decorator
    def test(self):
        self.detect_and_click("tili", self.perform_click_center, stop_if_no_detect=True)

    @time_and_game_limit_decorator
    def test2(self):
        self.detect_and_click("gold", self.perform_click_center, stop_if_no_detect=True)
