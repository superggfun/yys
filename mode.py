"""
Mode模块，提供了Mode类，它是WindowDetect类的子类，实现了特定的点击操作。
"""

import time, random

from window_detect import WindowDetect

class Mode(WindowDetect):
    """
    Mode类，WindowDetect的子类，实现了特定的点击操作。
    """
    def __init__(self, window_name, use_sct=True):
        super().__init__(window_name, use_sct)

    def yuhun(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        while True:  # 无限循环，持续进行对象检测和点击操作
            self.detect_and_click("win", self.perform_click_center)
            self.detect_and_click("hun", self.perform_click_all)
            self.detect_and_click("tiaozhan_on", self.perform_click_center)

    def tupo(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        while True:  # 无限循环，持续进行对象检测和点击操作
            self.detect_and_click("tupo_new", self.perform_click_center)
            self.detect_and_click("jingong", self.perform_click_center)
            self.detect_and_click("zhunbei", self.perform_click_center)
            self.detect_and_click_any(["tupo_fail", "hun"], self.perform_click_all)
            time.sleep(1)
            self.detect_and_click("hun", self.perform_click_all, 1)

    def fuben(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        while True:  # 无限循环，持续进行对象检测和点击操作
            self.detect_and_click("instances_28", self.perform_click_center, 3)
            print(1)
            self.detect_and_click("tansuo_botton", self.perform_click_center)
            print(2)
            while True:  # 内部循环，处理 "attack_head" 和 "attack"
                print(3)
                #self.swipe_screen()
                name = self.detect_and_click_priority({"attack_head": 2, "attack": 1}, self.perform_click_center, no_delay=True)
                if name is None:  # 添加了错误处理，如果detect_and_click_priority函数返回None（可能是超时或者出错），那么打印一个消息并跳出内部循环
                    print("No Detection!")
                    self.swipe_screen()
                    continue
                #if self.detect_and_click("zhunbei", self.perform_click_center, 5):
                #    print(7)
                #    continue
                if not self.detect_and_click("hun", self.perform_click_all, 30):
                    continue
                print(4)
                if name == "attack_head":
                    print(5)
                    time.sleep(random.uniform(1, 2))
                    for _ in range(4):
                        self.detect_and_click("gift", self.perform_click_center, 1)
                        self.detect_and_click("gain_gift", self.perform_click_all, 1)
                    break  # 如果检测到 "attack_head"，则跳出内部循环
                print(6)
                time.sleep(random.uniform(1, 2))  # 在循环结束时暂停，以避免过快的循环


    def t(self):
        while True:  # 无限循环，持续进行对象检测和点击操作
            self.detect_and_click("instances_27", self.perform_click_center)

