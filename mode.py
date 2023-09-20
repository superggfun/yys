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
                print("[停止] 已达到时间限制，停止游戏。")
                self.stop()
                break
            if self.game_limit is not None and game_count >= self.game_limit:
                print("[停止] 已达到游戏次数限制，停止游戏。")
                self.stop()
                break
            if not self.running:
                break
            
            try:
                func(self, *args, **kwargs)
                game_count += 1
                print(f"[通告] 已完成游戏次数：{game_count}次。")  # 打印已完成的游戏次数
            except Exception as e:
                print(f"发生错误：{e}")
                self.stop()
                break
    return wrapped_function


class Mode(ObjectInteractor):
    """
    Mode类，ObjectInteractor的子类，实现了特定的点击操作。
    """
    def __init__(self, window_name: str, mode: Literal['mss', 'win_api', 'adb'] = 'mss', game_limit=None, time_limit=None, stop_callback=None):
        super().__init__(window_name, mode, stop_callback=stop_callback)
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
        self.detect_and_click("instances_14", self.perform_click_center, 3)
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
                self.just_click(self.perform_click_center_lower_right)
                continue
            #if self.detect_and_click("zhunbei", self.perform_click_center, 5):
            #    print(7)
            #    continue
            if not self.detect_and_click("win", self.perform_click_all, 25):
                continue
            self.detect_and_click("hun", self.perform_click_all, 2)
            print(4)
            time.sleep(1)
            if name == "attack_head":
                print(5)
                time.sleep(2)
                for _ in range(4):
                    self.detect_and_click("gift", self.perform_click_center, 1)
                    self.detect_and_click("gain_gift", self.perform_click_all, 1)
                break  # 如果检测到 "attack_head"，则跳出内部循环
        self.detect_and_click("baoxiang", self.perform_click_center, 2)


    @time_and_game_limit_decorator
    def test(self):
        self.detect_and_click("tili", self.perform_click_center)

    @time_and_game_limit_decorator
    def test2(self):
        self.detect_and_click("gold", self.perform_click_center)

    @time_and_game_limit_decorator
    def huodong(self):
        self.detect_and_click("huodong_tiaozhan", self.perform_click_center)
        self.detect_and_click("gain_gift", self.perform_click_center_lower)
        random_sleep_time = random.uniform(0.4, 0.7)
        time.sleep(random_sleep_time)
        self.just_click(self.perform_click_center_lower)

    @time_and_game_limit_decorator
    def tupo2(self):
        while not self.detect_and_click("tupo_new", self.perform_click_center, timeout = 1):
            print("没有检测到目标，进行滑动")
            self.swipe_screen(direction="up")

        print("检测到目标并点击了，现在执行后续操作")
        print(444)
        
        entered_while = False
        while self.detect_and_click("jingong_off", self.perform_click_center, timeout = 1.5):
            entered_while = True
            self.detect_and_click("tupo_new", self.perform_click_center)
            print("[通知]等待5秒")
            time.sleep(5)
            self.detect_and_click("tupo_new", self.perform_click_center)
        print(555)
        #if entered_while:
            #self.detect_and_click("tupo_new", self.perform_click_center)
        time.sleep(1)
        self.detect_and_click("jingong", self.perform_click_center)
        self.detect_and_click("zhunbei", self.perform_click_center)
        self.detect_and_click_any(["tupo_fail", "hun"], self.perform_click_all)
        time.sleep(1)
        self.detect_and_click("hun", self.perform_click_all, 1)

    @time_and_game_limit_decorator
    def test(self):
        self.just_click(self.perform_click_center_lower_right)
        time.sleep(5)