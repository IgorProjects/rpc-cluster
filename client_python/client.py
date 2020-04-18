import sys
from decimal import Decimal
from threading import Thread

import grpc
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import cluster_pb2
import cluster_pb2_grpc
from main_window import Ui_main_window


class Application:
    def __init__(self, argv):
        self.stub = None
        self.channel = None
        self.plot_data = []
        self.is_drew = True

        self.app = QApplication(argv)
        main_window = QMainWindow()
        self.ui = Ui_main_window()
        self.ui.setupUi(main_window)

        canvas = FigureCanvas(Figure())
        self.ui.plot_container.addWidget(NavigationToolbar(canvas, main_window))
        self.ui.plot_container.addWidget(canvas)
        self.dynamic_ax = canvas.figure.gca(projection='3d')
        main_window.show()

        self.ui.start_button.clicked.connect(self.start_task_handler)
        self.ui.suspend_button.clicked.connect(self.suspend_resume_task_handler)
        self.ui.terminate_button.clicked.connect(self.terminate_task_handler)

    def __del__(self):
        if self.channel is not None:
            self.channel.close()

    def start_task_handler(self):
        if self.channel is not None:
            self.channel.close()
        self.channel = grpc.insecure_channel(f'{self.ui.ip_edit.text()}:{self.ui.port_edit.text()}')
        self.stub = cluster_pb2_grpc.ControlServiceStub(self.channel)

        run_message = cluster_pb2.RunMessage(function_name=self.ui.path_edit.text())
        for row in range(self.ui.param_table.rowCount()):
            run_message.param.append(Decimal(self.ui.param_table.item(row, 1).text()))
        Thread(target=self.read_points, args=(run_message,)).start()

    def read_points(self, run_message):
        for point_batch in self.stub.StartTask(run_message):
            self.plot_data.append(zip(*((point.x, point.y, point.z) for point in point_batch.point)))
            if self.is_drew:
                self.is_drew = False
                Thread(target=self.draw).start()

    def draw(self):
        for x, y, z in self.plot_data:
            self.dynamic_ax.clear()
            self.dynamic_ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0)
            self.dynamic_ax.figure.canvas.draw()
        self.is_drew = True

    def suspend_resume_task_handler(self):
        if self.ui.suspend_button.text() != self.ui.suspend_button.SUSPEND_STATE:  # because state is changed
            self.stub.SuspendTask(cluster_pb2.Empty())
        else:
            self.stub.ResumeTask(cluster_pb2.Empty())

    def read_current_points_handler(self):
        pass

    def terminate_task_handler(self):
        self.stub.TerminateTask(cluster_pb2.Empty())

    def start(self):
        sys.exit(self.app.exec_())


app = Application(sys.argv)
app.start()
