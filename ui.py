"""
该模块提供一个PyQt5应用，该应用可以在选项卡界面中管理多个"ClickerPage"实例。
"""

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                            QTabWidget, QWidget, QGroupBox, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from childpage import ClickerPage

class ClickerApp(QMainWindow):
    """
    ClickerApp的主应用类。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Alphay')
        self.setGeometry(100, 100, 480, 320)
        self.setFixedSize(480, 469)

        self.setWindowIcon(QIcon('AY.ico'))
        

        self.add_page_button = QPushButton("+")
        self.add_page_button.clicked.connect(self.add_page)
        self.remove_page_button = QPushButton("-")
        self.remove_page_button.clicked.connect(self.remove_page)

        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout.addWidget(self.add_page_button)
        self.buttons_layout.addWidget(self.remove_page_button)

        self.tab_widget = QTabWidget()
        self.tab_widget.setCornerWidget(self.buttons_widget, Qt.TopRightCorner)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tab_widget)

        self.group_box = QGroupBox("账号")
        self.group_box.setLayout(self.layout)

        self.setCentralWidget(self.group_box)

        self.add_page()

        self.show()

        self.apply_qss()

    def apply_qss(self):
        """
        从外部.qss文件应用QSS样式到应用程序。
        """
        with open('style.qss', 'r', encoding='utf-8') as style_file:  # specify the encoding
            self.setStyleSheet(style_file.read())

    def add_page(self):
        """
        在QTabWidget中添加一个新的"ClickerPage"。
        """
        page = ClickerPage()
        self.tab_widget.addTab(page, f"账号 {self.tab_widget.count() + 1}")

    def remove_page(self):
        """
        如果有多个页面，则从QTabWidget中移除当前的"ClickerPage"。
        """
        if self.tab_widget.currentIndex() != 0 and self.tab_widget.count() > 1:
            self.tab_widget.removeTab(self.tab_widget.currentIndex())

if __name__ == '__main__':
    app = QApplication([])
    ex = ClickerApp()
    app.exec_()
