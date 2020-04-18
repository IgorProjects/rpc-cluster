from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QPushButton


class SuspendButton(QPushButton):
    @pyqtSlot(name='switchState')
    def switch_state(self):
        self.setText("Продолжить" if self.text() == "Приостановить" else "Приостановить")

    @pyqtSlot(name='setDefaultState')
    def set_default_state(self):
        self.setText("Приостановить")
