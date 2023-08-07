from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QGridLayout, QComboBox,
                             QLineEdit, QPushButton, QSizePolicy, QHBoxLayout, QLabel, QTextBrowser, QMessageBox)

from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread

from win32gui import EnumWindows, GetWindowText, IsWindow, IsWindowEnabled, IsWindowVisible

from mode import Mode

import re, sys, torch

class RedirectStdout:
    def __init__(self, new_stdout):
        self.new_stdout = new_stdout
        self.old_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self.new_stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

class Stream(QObject):
    newText = pyqtSignal(str, str)  # 添加一个参数传递页面标识符

    def __init__(self, page_id=None):
        super().__init__()
        self.page_id = page_id

    color_mapping = {
        "[通告]": ("[通告]", "orange"),
        "An error occurred": ("[错误]", "red"),
        "发生错误": ("[错误]", "red"),
        "[错误]": ("[错误]", "red"),
        "[停止]": ("[停止]", "red"),
        "[识别]": ("[识别]", "green")
    }

    def write(self, text):
        #self.newText.emit(str(text) + '<br />')
        for phrase, (replacement, color) in self.color_mapping.items():
            if text.startswith(phrase) and self.page_id:
                text = text.replace(phrase, f'<font color="{color}">{replacement}</font>')
                self.newText.emit(self.page_id, str(text) + '<br />')  # 将标识符作为参数发送

    def flush(self):
        pass

class Worker(QThread):
    signal = pyqtSignal(str)

    def __init__(self, parent=None, func=None, args=(), stream=None):
        QThread.__init__(self, parent)
        self.func = func
        self.args = args
        self.stream = stream

    def run(self):
        try:
            # 使用RedirectStdout来捕获在func中的print语句
            with RedirectStdout(self.stream):
                result = self.func(*self.args)

            # 在调用func之后，直接将result写入stream
            if result is not None:
                self.stream.write(str(result))
        except Exception as e:
            self.signal.emit(f"Error: {e}")

    def stop(self):
        self.running = False  # 停止 run 方法的执行

    def print(self, message):
        # 一个用于代替 print 的方法
        self.signal.emit(message)

class ClickerPage(QWidget):
    """
    ClickerPage类，这是我们应用的主界面。
    """
    
    def __init__(self, port="", page_id=None):
        super().__init__()
        self.page_id=str(page_id)
        self.port = port if port is not None else ""  # 如果端口是None，那么设置为空字符串
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
        self.port_input.setText(str(port))
        self.port_input.textChanged.connect(self.port_changed)  # 添加此行

        self.confirm_button = QPushButton("保存")  # 确定按钮
        self.confirm_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.confirm_button.clicked.connect(self.save_port)

        self.capture_mode_layout.addWidget(self.window_selector, 0, 1)
        self.capture_mode_layout.addWidget(self.port_input, 0, 1)
        self.capture_mode_layout.addWidget(self.confirm_button, 0, 2)

        # 功能选项
        self.feature_box = QGroupBox("Feature")
        self.feature_layout = QVBoxLayout()
        self.feature_box.setLayout(self.feature_layout)

        self.features = ["御魂司机", "御魂跟车", "结界突破", "困28", "痴", "契灵", "测试体力", "测试金币"]
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

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self.log(f'<font color="green">连接 {device_name} 成功</font><br />')
        else:
            self.log('<font color="red">未连接到GPU设备，正在使用CPU</font><br />')



        # 保存旧的 sys.stdout
        #self.old_stdout = sys.stdout

        # 创建一个 Stream 实例并将其设置为 sys.stdout
        #self.page_id = str(uuid.uuid4())  # 生成唯一标识符
        self.stream = Stream(page_id=self.page_id)
        #self.stream = Stream()
        #sys.stdout = self.stream

        # 连接 Stream 的 newText 信号到更新 log_display 的方法
        self.stream.newText.connect(self.updateText)

        # 显式调用 change_capture_mode 和 change_stop_options，确保捕获模式和停止选项的状态正确
        self.change_capture_mode(self.capture_mode_combo.currentText())
        self.change_stop_options(self.stop_options_combo.currentText())

        # 版本信息
        self.version_label = QLabel("Version: 5.0")
        self.layout.addWidget(self.version_label)

        # 初始化状态
        self.set_state("initialized")

        #self.log("欢迎使用AlphaY")

        # 在类的初始化函数中
        self.mode = None

    def save_port(self):
        text = self.port_input.text()
        
        # 检查字符串是否匹配端口格式
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}$', text):
            self.port = text
        # 检查字符串是否匹配ADB设备号格式
        elif re.match(r'^[A-Z0-9]{16}$', text):
            self.port = text
        else:       
            QMessageBox.warning(self, '无效的输入', '输入必须是IP:端口或者ADB设备号。')
            return  # 如果输入无效，则不保存端口
        
        main_window = self.window()  # 获取主窗口
        main_window.save_ports()  # 直接调用 save_ports 方法

    # 添加此函数来处理用户更改端口的情况
    def port_changed(self, new_port):
        self.port = new_port


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
        # 创建 Worker 线程
        self.worker = Worker(self, self._init_mode_and_run, 
                             (self.get_mode_value(), mode_value, game_limit, time_limit, selected_feature),
                             stream=self.stream)
        #self.worker.signal.connect(self.log_display.append)
        self.worker.start()  # 开始运行 worker 线程

    def _init_mode_and_run(self, window_name, mode_value, game_limit, time_limit, mode_method_name):
        self.mode = Mode(window_name, mode=mode_value, game_limit=game_limit, time_limit=time_limit)
        
        self.feature_dict = {
            "御魂司机": self.mode.yuhun, 
            "御魂跟车": self.mode.yuhun2,
            "结界突破": self.mode.tupo, 
            "痴": self.mode.chi,
            "契灵": self.mode.qiling,
            "测试体力": self.mode.test,
            "测试金币": self.mode.test2,
            "困28": self.mode.fuben
        }

        
        mode_method = self.feature_dict[mode_method_name]
        mode_method()

    def set_state_to_initialized(self):
        self.set_state("initialized")

        
    def stop_clicked(self):
        """
        "停止"按钮的点击事件处理。
        """
        if self.mode is None:
            return
        self.set_state("initialized")
        self.mode.stop()  # 注意这里使用的是self.mode
        self.worker.stop()  # 停止 Worker 线程

    def pause_continue_clicked(self):
        """
        "暂停/继续"按钮的点击事件处理。
        """
        if self.mode is None:
            return
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
    
    def updateText(self, page_id, text):
        if page_id == self.page_id:
            cursor = self.log_display.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertHtml(text)
            self.log_display.setTextCursor(cursor)

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
