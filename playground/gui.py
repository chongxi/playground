import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QWidget, QTextBrowser, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout

import time
from datetime import datetime

import torch as torch
from torch.multiprocessing import Process, Pipe

#---------new module---------
from base import Jovian
from view import maze_view
from utils import Timer

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
maze_file       = dir_path+'/base/maze/obj/maze_2d.obj'
maze_coord_file = dir_path+'/base/maze/2dmaze_2cue_follow1_4.coords'
cue1_file       = dir_path+'/base/maze/obj/constraint_cue.obj'
cue0_file       = dir_path+'/base/maze/obj/goal_cue.obj'


def _jovian_process(pipe, jov):
    while True:
        with Timer('', verbose=True):
            _t, _coord = jov.readline().parse()
            pipe.send((_t, _coord))
        print(_t, _coord)


class play_GUI(QWidget):
    """
    GUI for experiment: control file, task parameter; navigation visualization, 
    """
    def __init__(self, arg):
        super(play_GUI, self).__init__()
        self.arg = arg
        self.ts  = np.array([])
        self.pos = np.array([])
        self.event_log = {}

        self.pipe_jovian_side, self.pipe_gui_side = Pipe()
        self.pipe_timer = QtCore.QTimer(self)
        self.pipe_timer.timeout.connect(self.vr_parsing_protocol)
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
        self.vrBtn.toggled.connect(self.jovian_process_toggle)

        BtnLayout = QGridLayout()
        BtnLayout.addWidget(self.vrBtn,0,0)

        #4. TextBrowser
        self.TextBrowser = QTextBrowser()
        self.TextBrowser.setGeometry(40, 90, 180, 79)

        #4. Navigation view for both viz and interaction 
        self.nav_view = maze_view()
        self.nav_view.load_maze(maze_file = maze_file, 
                                maze_coord_file = maze_coord_file) 
        self.nav_view.load_cue(cue_file = cue0_file, cue_name = '_dcue_000')
        self.nav_view.load_cue(cue_file = cue1_file, cue_name = '_dcue_001')

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

    def jovian_process_toggle(self, checked):
        if checked:
            self.jovian_process_start()
        else:
            self.jovian_process_stop()

    def jovian_process_start(self):
        self.jov = Jovian()
        self.vrBtn.setText('VR Stream ON')
        self.vrBtn.setStyleSheet("background-color: green")
        self.pipe_timer.start()
        self.jovian_process = Process(target=_jovian_process, args=(self.pipe_jovian_side, self.jov))
        self.jovian_process.daemon = True
        self.jovian_process.start()        
        self.nav_view.connect(self.jov)

    def jovian_process_stop(self):
        # self.stop_Socket()
        self.vrBtn.setText('VR Stream Off')
        self.vrBtn.setStyleSheet("background-color: white")
        self.jovian_process.terminate()
        self.jovian_process.join()
        self.pipe_timer.stop()
        self.ac_tag = 0
        self.jov.reset()


    def vr_parsing_protocol(self):
        ts, coord = self.pipe_gui_side.recv()
        self.nav_view.current_pos = np.array(coord)


    def Behavior_Protocol(self):
        print(self._time, self._coord)  

