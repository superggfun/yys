from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QGridLayout, QComboBox,
                             QLineEdit, QPushButton, QSizePolicy, QHBoxLayout, QLabel, QTextBrowser, QMessageBox)

from PyQt5.QtCore import Qt

from win32gui import EnumWindows, GetWindowText, IsWindow, IsWindowEnabled, IsWindowVisible

import threading
from mode import Mode

class ClickerPage(QWidget):
    """
    ClickerPage类，这是我们应用的主界面。
    """
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        # 使用静态方法的方式
        titles_to_find = ['多屏协同', '模拟器', '阴阳师-网易游戏', 'miui+', 'VM VirtualBox']
        self.screen_list = ClickerPage.get_window_handles(titles_to_find)

        # 截图模式
        self.capture_mode_box = QGroupBox("Capture Mode")
        self.capture_mode_layout = QGridLayout()
        self.capture_mode_box.setLayout(self.capture_mode_layout)

        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.setFixedWidth(150)
        self.capture_mode_combo.addItems(["MSS Capture", "WinAPI Capture", "ADB Capture"])
        self.capture_mode_combo.currentTextChanged.connect(self.change_capture_mode)
        self.capture_mode_layout.addWidget(self.capture_mode_combo, 0, 0)

        self.window_selector = QComboBox()  # 窗口选择下拉菜单
        window_titles = [item[1] for item in self.screen_list]  # 从screen_list中提取窗口标题
        self.window_selector.addItems(window_titles)  # 将窗口标题添加到下拉菜单中
        self.port_input = QLineEdit()  # 端口输入
        self.confirm_button = QPushButton("Confirm")  # 确定按钮
        self.confirm_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.capture_mode_layout.addWidget(self.window_selector, 0, 1)
        self.capture_mode_layout.addWidget(self.port_input, 0, 1)
        self.capture_mode_layout.addWidget(self.confirm_button, 0, 2)

        # 功能选项
        self.feature_box = QGroupBox("Feature")
        self.feature_layout = QVBoxLayout()
        self.feature_box.setLayout(self.feature_layout)

        self.features = ["御魂", "结界突破", "痴","测试体力", "测试金币"]
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(self.features)
        self.feature_layout.addWidget(self.feature_combo)

        self.layout.addWidget(self.capture_mode_box)
        self.layout.addWidget(self.feature_box)

        # 控制按钮和选项
        self.control_option_box = QGroupBox("Control and Options")
        self.control_option_layout = QVBoxLayout()
        self.control_option_box.setLayout(self.control_option_layout)

        self.control_button_layout = QHBoxLayout()
        self.pause_continue_button = QPushButton("暂停/继续")
        self.pause_continue_button.setObjectName("PC")
        self.start_button = QPushButton("开始")
        self.start_button.setObjectName("startButton")
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("StopButton")
        self.control_buttons = [self.pause_continue_button, self.start_button, self.stop_button]
        self.start_button.clicked.connect(self.start_clicked)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.pause_continue_button.clicked.connect(self.pause_continue_clicked)
        for button in self.control_buttons:
            button.setMinimumHeight(50)
            self.control_button_layout.addWidget(button)
        self.control_option_layout.addLayout(self.control_button_layout)

        self.stop_options_combo = QComboBox()
        self.stop_options_combo.addItems(["No limit", "Number of Games", "Timer"])
        self.stop_options_combo.currentTextChanged.connect(self.change_stop_options)
        self.stop_option_input = QLineEdit()  # 用于输入游戏数量或时间
        self.stop_option_input.setVisible(False)  # 默认情况下隐藏
        

        self.stop_options_layout = QHBoxLayout()
        self.stop_options_layout.addWidget(self.stop_options_combo)
        self.stop_options_layout.addWidget(self.stop_option_input)
        self.control_option_layout.addLayout(self.stop_options_layout)

        self.layout.addWidget(self.control_option_box)

        # 添加日志显示
        self.log_display = QTextBrowser()
        self.log_display.setObjectName("textBrowser")
        self.log_display.setOpenExternalLinks(True)  # 允许打开外部链接
        self.log_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # 始终显示垂直滚动条
        self.layout.addWidget(self.log_display)

        # 显式调用 change_capture_mode 和 change_stop_options，确保捕获模式和停止选项的状态正确
        self.change_capture_mode(self.capture_mode_combo.currentText())
        self.change_stop_options(self.stop_options_combo.currentText())

        # 版本信息
        self.version_label = QLabel("Version: 5.0")
        self.layout.addWidget(self.version_label)

        # 初始化状态
        self.set_state("initialized")

        self.log("欢迎使用AlphaY")


    def start_clicked(self):
        """
        "开始"按钮的点击事件处理。
        """
        
        self.set_state("running")
        # 创建Mode实例
        capture_mode_dict = {
            "MSS Capture": "mss",
            "WinAPI Capture": "winapi",
            "ADB Capture": "adb"
        }
        mode_value = capture_mode_dict[self.capture_mode_combo.currentText()]

        # 从GUI中获取限制参数
        stop_option = self.stop_options_combo.currentText()

        if stop_option == "No limit":
            game_limit = None
            time_limit = None
        elif stop_option == "Number of Games":
            try:
                limit_value = int(self.stop_option_input.text())
                game_limit = limit_value
                time_limit = None
            except ValueError:
                QMessageBox.warning(self, "Invalid input", "Please enter a valid number for limit.")
                return
        elif stop_option == "Timer":
            try:
                limit_value = int(self.stop_option_input.text())
                game_limit = None
                time_limit = limit_value
            except ValueError:
                QMessageBox.warning(self, "Invalid input", "Please enter a valid number for limit.")
                return

        
        # 创建线程
        selected_feature = self.feature_combo.currentText()
        self.thread1 = threading.Thread(target=self._init_mode_and_run, 
                                    args=(self.get_mode_value(), mode_value, game_limit, time_limit, selected_feature),
                                    daemon=True)
        self.thread1.start()

    def _init_mode_and_run(self, window_name, mode_value, game_limit, time_limit, mode_method_name):
        self.mode = Mode(window_name, mode=mode_value, game_limit=game_limit, time_limit=time_limit)
        
        self.feature_dict = {
            "御魂": self.mode.yuhun, 
            "结界突破": self.mode.tupo, 
            "痴": self.mode.chi,
            "测试体力": self.mode.test,
            "测试金币": self.mode.test2
        }
        
        mode_method = self.feature_dict[mode_method_name]
        mode_method()

    def stop_clicked(self):
        """
        "停止"按钮的点击事件处理。
        """
        self.set_state("initialized")
        self.mode.stop()  # 注意这里使用的是self.mode

    def pause_continue_clicked(self):
        """
        "暂停/继续"按钮的点击事件处理。
        """
        if self.pause_continue_button.text() == "暂停":
            self.set_state("paused")
            self.mode.pause()  # 注意这里使用的是self.mode
        else:
            self.set_state("running")
            self.mode.resume()  # 注意这里使用的是self.mode

    def set_state(self, state):
        """
        根据给定的状态设置按钮的状态。
        """
        if state == "initialized":
            self.start_button.setEnabled(True)
            self.pause_continue_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.pause_continue_button.setText("暂停/继续")
        elif state == "running":
            self.start_button.setEnabled(False)
            self.pause_continue_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.pause_continue_button.setText("暂停")
        elif state == "paused":
            self.start_button.setEnabled(False)
            self.pause_continue_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.pause_continue_button.setText("继续")

    def change_capture_mode(self, mode):
        """
        根据选择的捕获模式更改相关控件的可见性。
        """
        self.window_selector.setVisible(mode != "ADB Capture")
        self.port_input.setVisible(mode == "ADB Capture")
        self.confirm_button.setVisible(mode == "ADB Capture")

    def change_stop_options(self, option):
        """
        根据选择的停止选项更改相关控件的可见性。
        """
        self.stop_option_input.setVisible(option != "No limit")

    # 使用 log_display 代替 print 函数
    def log(self, message):
        """
        使用log_display代替print函数。
        """
        self.log_display.append(message)

    def get_mode_value(self):
        selected_capture_mode = self.capture_mode_combo.currentText()
        if selected_capture_mode == "ADB Capture":
            return self.port_input.text()
        else:
            selected_window_title = self.window_selector.currentText()
            # 通过标题找到对应的窗口句柄
            for item in self.screen_list:
                if item[1] == selected_window_title:
                    return item[0]
        return None

    @staticmethod
    def get_window_handles(titles_to_find):
        """
        遍历所有窗口并返回包含特定标题的窗口的句柄列表
        :param titles_to_find: 查找的窗口标题列表
        :return: 包含特定标题的窗口的句柄列表
        """

        hwnd_title = {}

        def get_all_hwnd(hwnd, mouse):
            """
            EnumWindows的回调函数，检查每个窗口，并将标题和句柄添加到hwnd_title字典中
            """
            if IsWindow(hwnd) and IsWindowEnabled(hwnd) and IsWindowVisible(hwnd):
                hwnd_title.update({hwnd: GetWindowText(hwnd)})

        EnumWindows(get_all_hwnd, 0)
        
        return [[h, t] for h, t in hwnd_title.items() if any(title in t for title in titles_to_find)]
