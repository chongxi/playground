import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QWidget, QSplitter, QComboBox, QTextBrowser, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
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


class play_GUI(QWidget):
    """
    GUI for experiment: control file, task parameter; navigation visualization, 
    """
    def __init__(self):
        # super(play_GUI, self).__init__()
        QWidget.__init__(self)
        self.event_log = {}
        self.nav_view_timer = QtCore.QTimer(self)
        self.nav_view_timer.timeout.connect(self.nav_view_update)
        self.init_UI()

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
        DirLayout.addWidget(self.mzBtn,   1,0,1,1)
        DirLayout.addWidget(self.combo,   1,1,1,1)

        #2. File name and Layout
        self.Year_Date_Time = datetime.now().strftime("%Y%m%d_%H%M")

        #3. Bottons
        self.vrBtn = QPushButton("VR Stream Off",self)
        self.vrBtn.setCheckable(True)
        self.vrBtn.setStyleSheet("background-color: darkgrey")
        self.vrBtn.toggled.connect(self.jovian_process_toggle)

        BtnLayout = QGridLayout()
        BtnLayout.addWidget(self.vrBtn,0,0)

        #4 Reward Parameter
        self.reward_time_label   = QLabel('Reward Time: 1s')
        self.touch_radius_label = QLabel('Reward Radius: 18')
        self.reward_time   = QSlider(Qt.Horizontal, self)
        self.reward_time.setValue(10)
        self.reward_time.valueChanged.connect(self.reward_time_changed)

        self.touch_radius = QSlider(Qt.Horizontal, self)
        self.touch_radius.setValue(20)
        self.touch_radius.valueChanged.connect(self.touch_radius_changed)

        ParaLayout = QGridLayout()
        ParaLayout.addWidget(self.reward_time_label,   0,0,1,1)
        ParaLayout.addWidget(self.reward_time,         0,1,1,1)
        ParaLayout.addWidget(self.touch_radius_label, 0,2,1,1)
        ParaLayout.addWidget(self.touch_radius,       0,3,1,1)

        #4. TextBrowser
        self.TextBrowser = QTextBrowser()
        self.TextBrowser.setGeometry(40, 90, 180, 79)

        #4. Navigation view for both viz and interaction 
        self.nav_view = maze_view()
        self._maze_loaded   = False
        self._task_selected = False

        #widget layout
        leftlayout = QVBoxLayout()
        leftlayout.addLayout(DirLayout)
        leftlayout.addLayout(BtnLayout)
        leftlayout.addWidget(self.TextBrowser)
        leftside = QWidget()
        leftside.setLayout(leftlayout)

        rightlayout = QVBoxLayout()
        rightlayout.addLayout(ParaLayout)
        rightlayout.addWidget(self.nav_view.native)
        rightside = QWidget()
        rightside.setLayout(rightlayout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(leftside)
        splitter.addWidget(rightside)        

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
        if self._maze_loaded:
            self.task_name = task_name
            # 1. Init Jovian and connect to maze navigation view 
            try:  # in cause it is already loaded 
                self.jov.pynq.shutdown(2)
            except:
                pass
            self.jov = Jovian()
            self.nav_view.connect(self.jov)  # shared cue_pos, shared tranformation
            
            # 2. Init Task
            try:
                self.task = globals()[self.task_name](self.jov)
                print('--------------------------')
                print('task: {}'.format(self.task_name))

                # 3. Task parameter
                self.task.reward_time = self.reward_time.value()/10. 
                self.jov.touch_radius = self.touch_radius.value()
                print('task reward time: {}, task touch radius: {}'.format(self.task.reward_time, self.jov.touch_radius))
                print('Task Ready')
                self._task_selected = True

            except:
                self.jov.pynq.shutdown(2)
                raise
        else:
            print('Load Maze folder first')


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
            print('load ', _cue_file)

        self._maze_loaded = True


    def reward_time_changed(self, value):
        if self._task_selected:
            self.reward_time_label.setText('Reward Time: {}'.format(str(value/10.)))
            self.task.reward_time = value/10. 
        else:
            print('select Task First')


    def touch_radius_changed(self, value):
        if self._task_selected:
            self.touch_radius_label.setText('Reward Radius: {}'.format(str(value)))
            self.jov.touch_radius = value
        else:
            print('select Task First')


    #------------------------------------------------------------------------------
    # jovian process (input, task fsm, output) in another CPU
    #------------------------------------------------------------------------------

    def jovian_process_toggle(self, checked):
        if checked:
            self.jovian_process_start()
        else:
            self.jovian_process_stop()


    def jovian_process_start(self):
        self.vrBtn.setText('VR Stream ON')
        self.vrBtn.setStyleSheet("background-color: green")
        self.jov.start()
        self.nav_view_timer.start(20)


    def jovian_process_stop(self):
        self.vrBtn.setText('VR Stream Off')
        self.vrBtn.setStyleSheet("background-color: darkgrey")
        self.jov.stop()
        self.nav_view_timer.stop()


    def nav_view_update(self):
        with Timer('', verbose=False):
            if self.jov.cnt>0:
                self.nav_view.current_pos = self.jov.current_pos.numpy()
                self.nav_view.cue_update()
