from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QTableWidget


class ParamTable(QTableWidget):
    @pyqtSlot(int, name='updateParams')
    def update_params(self, count):
        self.setRowCount(count)
