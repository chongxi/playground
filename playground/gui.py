import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, QEventLoop
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QWidget, QSplitter, QComboBox, QTextBrowser, QSlider, QPushButton, QTableWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QGridLayout
from PyQt5.QtGui import QIcon

import time
from datetime import datetime

#---------new module---------
from .base import Jovian
from .base import Fpga
from .base import task
from .base.task import one_cue_task, two_cue_task, one_cue_moving_task, JEDI, JUMPER, RING
from .view import maze_view
from .utils import Timer
from spiketag.view import probe_view, scatter_3d_view, raster_view

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '/base/maze/current/'


###################################################################################################################################################
# The Major GUI used for the real experiment
# raster: raster
###################################################################################################################################################

class play_raster_GUI(QWidget):
    """
    GUI for experiment: control file, task parameter; navigation visualization, 
    """
    def __init__(self, logger, bmi=None):
        # super(play_GUI, self).__init__()
        QWidget.__init__(self)
        self.log = logger
        self.current_group=0

        self.nav_view_timer = QtCore.QTimer(self)
        self.nav_view_timer.timeout.connect(self.nav_view_update)

        '''
        Setting bmi for jov, jov will emit `bmi_decode` event to the `task`
        '''
        if bmi is not None:
            self.bmi = bmi
            self.ras_view_timer = QtCore.QTimer(self)
            self.ras_view_timer.timeout.connect(self.ras_view_update)
            self.update_interval = 60
        else:
            self.bmi = None
            
        self.init_speed_thres = 1500
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

        #0. Set commonly used status variable (used by gui elements)
        self._maze_loaded   = False
        self._task_selected = False

        #1. Folder name (Maze files and task log) and layout
        self.DirName = QLineEdit(dir_path, self)
        self.DirName.returnPressed.connect(self.line_loadDialog)
        self.mzBtn = QPushButton("Load Maze",self)
        self.mzBtn.clicked.connect(self.btn_loadDialog)
        self.task_combo = QComboBox(self) 
        tasks = [x for x in dir(task) if '__' not in x and 'task' != x] # task is the module name
        for task_name in tasks:
            self.task_combo.addItem(task_name)
        self.task_combo.activated[str].connect(self.selectTask)

        DirLayout = QGridLayout()
        DirLayout.addWidget(self.DirName,    0,0,1,2)
        DirLayout.addWidget(self.mzBtn,      1,0,1,1)
        DirLayout.addWidget(self.task_combo, 1,1,1,1)

        #2. File name and Layout
        self.Year_Date_Time = datetime.now().strftime("%Y%m%d_%H%M")

        #3. Bottons
        self.vrBtn = QPushButton("VR Stream Off",self)
        self.vrBtn.setCheckable(True)
        self.vrBtn.setStyleSheet("background-color: darkgrey")
        self.vrBtn.toggled.connect(self.jovian_process_toggle)

        self.fpgaBtn = QPushButton("BMI Stream Off",self)
        self.fpgaBtn.setCheckable(True)
        self.fpgaBtn.setStyleSheet("background-color: darkgrey")
        self.fpgaBtn.toggled.connect(self.fpga_process_toggle)

        self.toggle_motion_Btn = QPushButton("Motion toggle", self)
        # self.toggle_motion_Btn.setCheckable(True)
        self.toggle_motion_Btn.setStyleSheet("background-color: darkgrey")
        # toggle_motion_Btn can only be connectted after jov init

        self.build_decoder_Btn = QPushButton("Build Decoder", self)
        self.build_decoder_Btn.setStyleSheet("background-color: darkgrey")
        self.build_decoder_Btn.clicked.connect(self.build_decoder)

        BtnLayout = QGridLayout()
        BtnLayout.addWidget(self.vrBtn,0,1)
        BtnLayout.addWidget(self.fpgaBtn,0,0)
        BtnLayout.addWidget(self.toggle_motion_Btn, 1,0)
        BtnLayout.addWidget(self.build_decoder_Btn, 1,1)

        #4 Reward Parameter
        ParaLayout = QGridLayout()

        self.reward_time_label   = QLabel('Reward Time: 1s')
        self.reward_time   = QSlider(Qt.Horizontal, self)
        self.reward_time.setValue(10)
        self.reward_time.valueChanged.connect(self.reward_time_changed)

        self.touch_radius_label = QLabel('Reward Radius: 18')
        self.touch_radius = QSlider(Qt.Horizontal, self)
        self.touch_radius.setValue(20)
        self.touch_radius.valueChanged.connect(self.touch_radius_changed)

        ParaLayout.addWidget(self.reward_time_label,   0,0,1,1)
        ParaLayout.addWidget(self.reward_time,         0,1,1,1)
        ParaLayout.addWidget(self.touch_radius_label,  0,2,1,1)
        ParaLayout.addWidget(self.touch_radius,        0,3,1,1)

        #5 BMI Parameter
        self.hd_window_label = QLabel('HD Window: 1s')
        self.hd_window = QSlider(Qt.Horizontal, self)
        self.hd_window.setMinimum(1)
        self.hd_window.setMaximum(16)
        self.hd_window.setSingleStep(1)
        self.hd_window.setValue(2) # 2/2 = 1 second
        self.hd_window.valueChanged.connect(self.hd_window_changed)

        self.bmi_teleport_radius_label = QLabel('speed thres')
        self.bmi_teleport_radius = QSlider(Qt.Horizontal, self)
        self.bmi_teleport_radius.setMinimum(0)
        self.bmi_teleport_radius.setMaximum(3000)
        self.bmi_teleport_radius.setSingleStep(1)        
        self.bmi_teleport_radius.setValue(self.init_speed_thres)
        self.bmi_teleport_radius.valueChanged.connect(self.bmi_teleport_radius_changed)

        ParaLayout.addWidget(self.hd_window_label,   0,4,1,1)
        ParaLayout.addWidget(self.hd_window,         0,5,1,1)
        ParaLayout.addWidget(self.bmi_teleport_radius_label,  0,6,1,1)
        ParaLayout.addWidget(self.bmi_teleport_radius,        0,7,1,1)

        #6. Raster View
        if self.bmi is not None:
            self.ras_view = raster_view(n_units=self.bmi.fpga.n_units+1, 
                                        t_window=5e-3, 
                                        view_window=10)
        else:
            self.ras_view = raster_view(n_units=100,
                                        t_window=5e-3,
                                        view_window=10)

        #7. Navigation view for both viz and interaction 
        self.nav_view = maze_view()

        #widget layout
        leftlayout = QVBoxLayout()
        leftlayout.addLayout(DirLayout)
        leftlayout.addLayout(BtnLayout)
        leftlayout.addWidget(self.ras_view.native)
        # leftlayout.addWidget(self.fet_view1.native) 
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
    def build_decoder(self):
        '''
        build decoder according to the task
        '''

#         if self.task_name == 'RING':
            # from spiketag.analysis.decoder import Maxout_ring
            # self.bmi.dec = Maxout_ring() 

        # if self.task_name == 'JEDI' or self.task_name == 'JUMPER':
        # else:
            # file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        spktag_file = str(QFileDialog.getOpenFileName(self, "load spktag", '../', '*.pd')[0])
        self.log.info('select   spktag {}'.format(spktag_file))
        pos_file = str(QFileDialog.getOpenFileName(self, "load saved position", '../', '(*.log);;(*.pd);;(*.bin)')[0])
        self.log.info('select position {}'.format(pos_file))
        from playground import build_decoder
        score = build_decoder(self.bmi, spktag_file, pos_file)
        self.log.info('BMI decoder params: {} decoding cells out of {} cells, {} t_step, {} t_window'.format(self.bmi.dec.neuron_idx.shape[0], 
                                                                                                             self.bmi.dec.fields.shape[0], 
                                                                                                             self.bmi.dec.t_step, 
                                                                                                             self.bmi.dec.t_window))
        self.log.info('BMI decoder R2-score (cross-validation enabled): {}'.format(score))
        self.log.info('BMI updating rule: {}'.format(self.bmi.bmi_update_rule))
        self.log.info('BMI posterior threshold: {}'.format(self.bmi.posterior_threshold))
        self.log.info('BMI position update buffer length: {}'.format(self.bmi.pos_buffer_len))

        # select task first
        if hasattr(self, 'jov'):
            self.jov.set_bmi(self.bmi, pos_buffer_len=self.bmi.pos_buffer_len)
        else:
            print('please select task first')




    def line_loadDialog(self):
        folder = self.DirName.text()
        self.load_maze(folder)


    def btn_loadDialog(self):
        folder = str(QFileDialog.getExistingDirectory(None))
        self.DirName.setText(folder)
        self.load_maze(folder)


    def selectTask(self, task_name):
        '''
        order from 1-5 is important, wrong order will cause crash.
        '''
        if self._maze_loaded:
            self.task_name = task_name
            # 1. Init Jovian 
            try:  # in cause it is already loaded 
                self.jov.pynq.shutdown(2)
            except:
                pass
            self.log.info('initiate Jovian and its socket connection')
            self.jov = Jovian()

            # 2. Init log and connect jov to maze navigation view, set counter cnt to 0
            self.jov.log = self.log
            self.jov.cnt.fill_(0)
            self.nav_view.connect(self.jov)  # shared cue_pos, shared tranformation
            self.jov.maze_border = self.maze_border
            self.toggle_motion_Btn.clicked.connect(self.jov.toggle_motion)
            # if the rotation encoder is not connected, don't show head direction arrow
            if self.jov.rot.is_connected is False:
                self.nav_view.arrow_len = 0

            # 3. Init Task
            try:
                self.task = globals()[self.task_name](self.jov)  # initialte task and pass jov into the task
                self.log.info('task: {}'.format(self.task_name))

                # 4. Task parameter
                ## reward time
                self.reward_time.setValue(self.task.reward_time*10) # define reward_time in each task
                self.reward_time_label.setText('Reward Time: {}'.format(str(self.task.reward_time)))
                ## reward radius
                self.jov.touch_radius.fill_(self.touch_radius.value())
                self.log.info('task reward time: {}, task touch radius: {}'.format(self.task.reward_time, self.jov.touch_radius))
                self.log.info('Task Ready')
                self._task_selected = True

            except:
                self.jov.pynq.shutdown(2)
                raise

            # 4. Init Speed threshold
            self.jov.bmi_teleport_radius.fill_(self.init_speed_thres)
        else:
            self.log.warn('Load Maze folder first')


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
        self.log.info('maze folder: {}'.format(folder))
        self.log.info('load {} {}'.format(maze_mesh_file, maze_coord_file))
        origin = np.array(self.nav_view.maze.coord['Origin']).astype(np.float32)
        border = np.array(self.nav_view.maze.coord['border']).astype(np.float32)
        self.log.info('maze_center: {},{}'.format(origin[0], origin[1]))   ## don't change the keyword maze_center, it is read building the decoder
        self.log.info('maze_border: {}'.format(border))                    ## don't change the keyword maze_border, it is read building the decoder
        self.maze_border = border.reshape(-1,2).T

        for file in cue_files:
            _cue_file = os.path.join(folder, file)
            self.nav_view.load_cue(cue_file=_cue_file, cue_name=file.split('.')[0])
            self.log.info('load {}'.format(_cue_file))

        self._maze_loaded = True

    #------------------------------------------------------------------------------
    # set slider for parameters
    # val/N is because slider only support integer number
    #------------------------------------------------------------------------------

    def reward_time_changed(self, value):
        if self._task_selected:
            self.reward_time_label.setText('Reward Time: {}'.format(str(value/10.)))
            self.task.reward_time = value/10. 
        else:
            self.log.warn('select Task First: Jovian initiate when selecting task')

    def touch_radius_changed(self, value):
        if self._task_selected:
            self.touch_radius_label.setText('Reward Radius: {}'.format(str(value)))
            self.jov.touch_radius.fill_(value)
        else:
            self.log.warn('select Task First: Jovian initiate when selecting task')

    def hd_window_changed(self, value):
        if self._task_selected:
            self.hd_window_label.setText('HD Window: {}'.format(str(value/2.)))
            self.jov.hd_window.fill_(value/2.)
        else:
            self.log.warn('select Task First: Jovian initiate when selecting task')

    def bmi_teleport_radius_changed(self, value):
        if self._task_selected:
            self.bmi_teleport_radius_label.setText('speed thres: {}'.format(value))
            self.jov.bmi_teleport_radius.fill_(value)
        else:
            self.log.warn('select Task First: Jovian initiate when selecting task')

    #------------------------------------------------------------------------------
    # jovian process (input, task fsm, output) in another CPU
    #------------------------------------------------------------------------------

    def jovian_process_toggle(self, checked):
        if self._task_selected:
            if checked:
                self.jovian_process_start()
                self.task_combo.setEnabled(False)
                self.DirName.setEnabled(False)
                self.mzBtn.setEnabled(False)
            else:
                self.jovian_process_stop()
                self.task_combo.setEnabled(True)
                self.DirName.setEnabled(True)
                self.mzBtn.setEnabled(True)
        else:
            self.log.warn('select Task First')


    def jovian_process_start(self):
        self.log.info('---------------------------------')
        self.log.info('jovian_process_start')
        self.log.info('---------------------------------')
        self.vrBtn.setText('VR Stream ON')
        self.vrBtn.setStyleSheet("background-color: green")
        self.jov.start()
        time.sleep(0.1)
        self.nav_view_timer.start(30)


    def jovian_process_stop(self):
        self.log.info('---------------------------------')
        self.log.info('jovian_process_stop')
        self.log.info('---------------------------------')
        self.vrBtn.setText('VR Stream Off')
        self.vrBtn.setStyleSheet("background-color: darkgrey")
        self.jov.stop()
        self.nav_view_timer.stop()


    def nav_view_update(self):
        # with Timer('', verbose=False):
        if self.jov.cnt>0:
            self.nav_view.current_pos = self.jov.current_pos.numpy()
            self.nav_view.current_hd  = self.jov.current_hd.numpy() 
            self.nav_view.cue_update()
        
            try:
                self.nav_view.posterior = self.jov.current_post_2d.numpy()
            except:
                pass

    #------------------------------------------------------------------------------
    # fpga process (input, task fsm, output) in another CPU
    #------------------------------------------------------------------------------

    def fpga_process_toggle(self, checked):
        if checked:
            self.fpga_process_start()
        else:
            self.fpga_process_stop()


    def fpga_process_start(self):
        self.log.info('---------------------------------')
        self.log.info('fpga_process_start')
        self.log.info('---------------------------------')
        self.fpgaBtn.setText('BMI Stream ON')
        self.fpgaBtn.setStyleSheet("background-color: green")
        self.bmi.start(gui_queue=False)
        self.ras_view_timer.start(self.update_interval)

    def fpga_process_stop(self):
        self.log.info('---------------------------------')
        self.log.info('fpga_process_stop')
        self.log.info('---------------------------------')
        self.fpgaBtn.setText('BMI Stream Off')
        self.fpgaBtn.setStyleSheet("background-color: darkgrey")
        self.bmi.stop()
        self.ras_view_timer.stop()


    def ras_view_update(self):
        self.ras_view.update_fromfile('./fet.bin', last_N=8000)
