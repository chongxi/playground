import sys
import socket

import numpy as np

# from spiketag.utils import Timer
# from spiketag.view import scatter_3d_view

from vispy import app, scene
from vispy.scene import visuals
from vispy.geometry import generation as gen, create_sphere

from PyQt5 import QtCore
from PyQt5.QtCore import QThread, QEventLoop
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QTextBrowser, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
from PyQt5.QtCore import Qt

from datetime import datetime
import time

import torch as torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe

#---------new module---------
from playground.base import Jovian
from playground.view import maze_view


def jovian_process(pipe, jov):
    while True:
        tic = time.time()
        _t, _coord = jov.readline().parse()
        toc = time.time()
        print((toc-tic)*1e3)
        print(_t, _coord)
        pipe.send((_t, _coord))


class BehavGUI(QWidget):
    """
    GUI for experiment: control file, task parameter; navigation visualization, 
    """
    def __init__(self, arg):
        super(BehavGUI, self).__init__()
        self.arg = arg
        self.ts  = np.array([])
        self.pos = np.array([])
        self.event_log = {}

        self.pipe_socket_side, self.pipe_gui_side = Pipe()
        self.pipe_timer = QtCore.QTimer(self)
        self.pipe_timer.timeout.connect(self.vr_parsing_protocol)
        self.jov = Jovian()
        self.initUI()

    def initUI(self, keys='interactive'):
        
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.darkGray)
        p.setColor(self.foregroundRole(), Qt.white)
        self.setPalette(p)

        #1. Folder name and layout
        self.DirLabel = QLabel("Folder Name", self)
        self.DirName = QLineEdit("~/Work/testbench", self)
        DirLayout = QHBoxLayout()
        DirLayout.addWidget(self.DirLabel)
        DirLayout.addWidget(self.DirName)

        #2. File name and Layout
        self.FileNameLabel = QLabel("File Name", self)
        self.FileName1 = QLineEdit("stFRVR", self)
        self.Year_Date_Time = datetime.now().strftime("%Y%m%d_%H%M")
        self.FileName2 = QLineEdit(self.Year_Date_Time,self)

        FileNameLayout = QHBoxLayout()
        FileNameLayout.addWidget(self.FileNameLabel)
        FileNameLayout.addWidget(self.FileName1)
        FileNameLayout.addWidget(self.FileName2)

        #3. Bottons
        self.vrBtn = QPushButton("VR Stream Off",self)
        self.vrBtn.setCheckable(True)
        self.vrBtn.setStyleSheet("background-color: white")
        self.vrBtn.toggled.connect(self.vr_stream_toggle)

        BtnLayout = QGridLayout()
        BtnLayout.addWidget(self.vrBtn,0,0)

        #4. TextBrowser
        self.TextBrowser = QTextBrowser()
        self.TextBrowser.setGeometry(40, 90, 180, 79)

        #4. Navigation view for both viz and interaction 
        self.nav_view = maze_view()
        self.nav_view.load_maze(maze_file='../playground/base/maze/obj/maze_2d.obj', 
                                maze_coord_file='../playground/base/maze/2dmaze_2cue_follow1_4.coords') 
        self.nav_view.load_cue(cue_file='../playground/base/maze/obj/constraint_cue.obj', 
                               cue_name='_dcue_001')
        self.nav_view.load_cue(cue_file='../playground/base/maze/obj/goal_cue.obj', 
                               cue_name='_dcue_000')
        self.nav_view.connect(self.jov.output)

        #widget layout
        WidLayout = QVBoxLayout()
        WidLayout.addLayout(DirLayout)
        WidLayout.addLayout(FileNameLayout)
        WidLayout.addLayout(BtnLayout)
        WidLayout.addWidget(self.TextBrowser)
        pLayout = QHBoxLayout()
        pLayout.addLayout(WidLayout)
        pLayout.addWidget(self.nav_view.native)

        self.setLayout(pLayout)

    def vr_stream_toggle(self, checked):
        if checked:
            self.vr_stream_start()
        else:
            self.vr_stream_stop()

    def vr_stream_start(self):
        self.vrBtn.setText('VR Stream ON')
        self.vrBtn.setStyleSheet("background-color: green")
        self.pipe_timer.start()
        self.vr_stream_process = Process(target=jovian_process, args=(self.pipe_socket_side, self.jov))
        self.vr_stream_process.daemon = True
        self.vr_stream_process.start()
        self.ac_tag = 1
        # for video recording
        # self.V_S.start()

    def vr_stream_stop(self):
        # self.stop_Socket()
        self.vrBtn.setText('VR Stream Off')
        self.vrBtn.setStyleSheet("background-color: white")
        self.vr_stream_process.terminate()
        self.vr_stream_process.join()
        self.pipe_timer.stop()
        self.ac_tag = 0

    def vr_parsing_protocol(self):
        # with Timer('parsing_protocol', verbose=False):
        # tic = time.time()
        ts, coord = self.pipe_gui_side.recv()
        # print(ts, coord)
        # toc = time.time()
        # elapse = (toc - tic)*1e3
        # if elapse>40:
        # print('elapse = ',elapse)
        # print(time, coord)
        # Task: budget cost < 2ms updating nav_view
        self.nav_view.current_pos = np.array(coord)

    def Behavior_Protocol(self):
        print(self._time, self._coord)  



if __name__ == '__main__':
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    shared_arr = torch.from_numpy(np.empty((10000, 2)))
    shared_arr.share_memory_()

    app = QApplication(sys.argv)
    gui = BehavGUI(0)
    # gui.setStyleSheet("background-color:#49C5B8;");
    gui.show()
    sys.exit(app.exec_())
