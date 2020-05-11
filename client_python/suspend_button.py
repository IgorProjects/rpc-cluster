from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QPushButton


class SuspendButton(QPushButton):
    SUSPEND_STATE = 'Приостановить'
    RESUME_STATE = 'Продолжить'

    @pyqtSlot(name='switchState')
    def switch_state(self):
        self.setText(self.SUSPEND_STATE if self.text() == self.RESUME_STATE else self.RESUME_STATE)

    @pyqtSlot(name='setDefaultState')
    def set_default_state(self):
        self.setText(self.SUSPEND_STATE)
