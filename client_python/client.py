import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from main_window import Ui_mainWindow

# little sample, I'll use it later
# self._dynamic_ax.clear()
# t = np.linspace(0, 10, 101)
# # Use fixed vertical limits to prevent autoscaling changing the scale
# # of the axis.
# self._dynamic_ax.set_ylim(-1.1, 1.1)
# # Shift the sinusoid as a function of time.
# self._dynamic_ax.plot(t, np.sin(t + time.time()))
# self._dynamic_ax.figure.canvas.draw()

app = QApplication(sys.argv)
mainWindow = QMainWindow()
ui = Ui_mainWindow()
ui.setupUi(mainWindow)

canvas = FigureCanvas(Figure(figsize=(5, 3)))
ui.plotContainer.addWidget(NavigationToolbar(canvas, mainWindow))
ui.plotContainer.addWidget(canvas)

static_ax = canvas.figure.subplots()
t = np.linspace(0, 10, 501)
static_ax.plot(t, np.tan(t), ".")

mainWindow.show()
sys.exit(app.exec_())
