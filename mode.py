"""
Mode模块，提供了Mode类，它是WindowDetect类的子类，实现了特定的点击操作。
"""

from window_detect import WindowDetect

class Mode(WindowDetect):
    """
    Mode类，WindowDetect的子类，实现了特定的点击操作。
    """
    def __init__(self, window_name):
        super().__init__(window_name)

    def click_operation(self):
        """
        执行点击操作。该方法首先进行物体检测，然后按照给定的类别顺序对检测到的物体进行点击操作。
        """
        while True:  # 无限循环，持续进行对象检测和点击操作
            self.detect_and_click("win", self.perform_click_center)
            self.detect_and_click("hun", self.perform_click_all)
