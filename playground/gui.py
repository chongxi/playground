import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QWidget, QSplitter, QComboBox, QTextBrowser, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
from PyQt5.QtGui import QIcon

import time
from datetime import datetime

#---------new module---------
from base import Jovian, one_cue_task, two_cue_task
from view import maze_view
from utils import Timer

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '/base/maze/obj/'
# maze_file       = dir_path+'/base/maze/obj/maze_2d.obj'
# maze_coord_file = dir_path+'/base/maze/obj/maze_2d.coords'
# cue1_file       = dir_path+'/base/maze/obj/_dcue_001.obj'
# cue0_file       = dir_path+'/base/maze/obj/_dcue_000.obj'


class play_GUI(QWidget):
    """
    GUI for experiment: control file, task parameter; navigation visualization, 
    """
    def __init__(self):
        # super(play_GUI, self).__init__()
        QWidget.__init__(self)
        # self.task_name = task_name
        self.event_log = {}
        self.nav_view_timer = QtCore.QTimer(self)
        self.nav_view_timer.timeout.connect(self.nav_view_update)
        self.init_UI()
        # self.init_Task()

    #------------------------------------------------------------------------------
    # gui layout
    #------------------------------------------------------------------------------

    def init_UI(self, keys='interactive'):
        
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.darkGray)
        p.setColor(self.foregroundRole(), Qt.white)
        self.setPalette(p)

        #1. Folder name (Maze files and task log) and layout
        self.DirName = QLineEdit(dir_path, self)
        self.DirName.returnPressed.connect(self.line_loadDialog)
        self.mzBtn = QPushButton("Load Maze",self)
        self.mzBtn.clicked.connect(self.btn_loadDialog)
        self.combo = QComboBox(self) 
        self.combo.addItem("one_cue_task")
        self.combo.addItem("two_cue_task")
        self.combo.activated[str].connect(self.selectTask)

        DirLayout = QGridLayout()
        DirLayout.addWidget(self.DirName, 0,0,1,2)
        DirLayout.addWidget(self.mzBtn, 1,0, 1, 1)
        DirLayout.addWidget(self.combo, 1,1 ,1, 1)

        #2. File name and Layout
        self.Year_Date_Time = datetime.now().strftime("%Y%m%d_%H%M")

        #3. Bottons
        self.vrBtn = QPushButton("VR Stream Off",self)
        self.vrBtn.setCheckable(True)
        self.vrBtn.setStyleSheet("background-color: darkgrey")
        self.vrBtn.toggled.connect(self.jovian_process_toggle)

        BtnLayout = QGridLayout()
        BtnLayout.addWidget(self.vrBtn,0,0)

        #4. TextBrowser
        self.TextBrowser = QTextBrowser()
        self.TextBrowser.setGeometry(40, 90, 180, 79)

        #4. Navigation view for both viz and interaction 
        self.nav_view = maze_view()

        #widget layout
        leftlayout = QVBoxLayout()
        leftlayout.addLayout(DirLayout)
        leftlayout.addLayout(BtnLayout)
        leftlayout.addWidget(self.TextBrowser)
        leftside = QWidget()
        leftside.setLayout(leftlayout)

        # rightlayout = QVBoxLayout()
        # rightlayout.addWidget(self.nav_view.native)
        # rightside = QWidget()
        # rightside.setLayout(rightlayout)
        # splitter = QSplitter(Qt.Horizontal)
        # splitter.addWidget(leftside)
        # splitter.addWidget(rightside)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(leftside)
        splitter.addWidget(self.nav_view.native)        

        pLayout = QHBoxLayout()
        pLayout.addWidget(splitter)
        self.setLayout(pLayout)

    #------------------------------------------------------------------------------
    # gui function
    #------------------------------------------------------------------------------
    def line_loadDialog(self):
        folder = self.DirName.text()
        self.load_maze(folder)

    def btn_loadDialog(self):
        folder = str(QFileDialog.getExistingDirectory(None))
        self.DirName.setText(folder)
        self.load_maze(folder)

    def selectTask(self, task_name):
        self.task_name = task_name

    # def init_Task(self):
        # 1. Init Jovian first 
        self.jov = Jovian()
        self.nav_view.connect(self.jov)  # shared cue_pos, shared tranformation

        # 2. Init Task
        self.task = globals()[self.task_name](self.jov)
        # self.task = two_cue_task(self.jov)
        print(self.task_name, self.task.state)


    def load_maze(self, folder):
        maze_files = [_ for _ in os.listdir(folder) if 'maze' in _]
        cue_files  = [_ for _ in os.listdir(folder) if 'cue'  in _]
        for file in maze_files:
            if file.endswith(".obj"):
                maze_mesh_file = os.path.join(folder, file)
            elif file.endswith(".coords"):
                maze_coord_file = os.path.join(folder, file)
        self.nav_view.load_maze(maze_file = maze_mesh_file, 
                                maze_coord_file = maze_coord_file) 
        self.nav_view.load_animal()
        print('load ', maze_mesh_file, maze_coord_file)

        for file in cue_files:
            _cue_file = os.path.join(folder, file)
            self.nav_view.load_cue(cue_file=_cue_file, cue_name=file.split('.')[0])
            # print(os.path.join(folder, file))
            print('load ', _cue_file)


    def jovian_process_toggle(self, checked):
        if checked:
            self.jovian_process_start()
        else:
            self.jovian_process_stop()


    def jovian_process_start(self):
        self.vrBtn.setText('VR Stream ON')
        self.vrBtn.setStyleSheet("background-color: green")
        self.jov.start()
        self.nav_view_timer.start()


    def jovian_process_stop(self):
        self.vrBtn.setText('VR Stream Off')
        self.vrBtn.setStyleSheet("background-color: darkgrey")
        self.jov.stop()
        self.nav_view_timer.stop()


    def nav_view_update(self):
        with Timer('', verbose=False):
            ts, coord = self.jov.get()
            self.nav_view.current_pos = coord
            self.nav_view.cue_update()
