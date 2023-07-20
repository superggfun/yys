"""
@superggfun
创建日期: 2023-07-18
版本: 1.1.0
描述: 基于Yolov5的阴阳师识别点击程序。
"""

import ctypes
from sys import executable
from mode import Mode

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

def run():
    """
    创建 Mode 实例并开始执行点击操作。
    """
    # Create an instance of ClickOperator
    window1 = Mode("阴阳师-网易游戏")

    # Call the click_operation method to start the click operations
    window1.click_operation()

if __name__ == '__main__':
    if is_admin():
        run()
    else:
        run_as_admin()
