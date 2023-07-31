"""
@superggfun
创建日期: 2023-07-18
版本: 1.2.1
描述: 基于Yolov5的阴阳师识别点击程序。
"""

import ctypes
from sys import executable
from mode import Mode
import threading

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


def start_mode_thread(window_name, method):
    mode = Mode(window_name, use_sct=True)
    thread = threading.Thread(target=method, args=(mode,), daemon=True)
    thread.start()
    return thread

def run():
    """
    创建 Mode 实例并开始执行点击操作。
    """
    thread1 = start_mode_thread("阴阳师-网易游戏", Mode.test)#阴阳师 - MuMu模拟器
    thread2 = start_mode_thread("阴阳师 - MuMu模拟器", Mode.test)#阴阳师-网易游戏
    
    thread1.join()
    thread2.join()
    
    #window1 = Mode("阴阳师-网易游戏",use_sct=False)
    #window1.test()



if __name__ == '__main__':
    #run()
    if is_admin():
        run()
    else:
        run_as_admin()
