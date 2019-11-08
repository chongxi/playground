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
from .base.task import one_cue_task, two_cue_task, one_cue_moving_task, JEDI, JUMPER
from .view import maze_view
from .utils import Timer
from spiketag.view import probe_view, scatter_3d_view

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += '/base/maze/complex/'


class play_GUI(QWidget):
    """
    GUI for experiment: control file, task parameter; navigation visualization, 
    """
    def __init__(self, logger, prb, bmi=None):
        # super(play_GUI, self).__init__()
        QWidget.__init__(self)
        self.log = logger
        self.current_group=0
        self.prb = prb
        @self.prb.connect
        def on_select(group_id, chs):
            self.current_group = group_id

        self.nav_view_timer = QtCore.QTimer(self)
        self.nav_view_timer.timeout.connect(self.nav_view_update)
        self.init_UI()

        '''
        temporary for JEDI and JUMPER experiment, needs to be refactored
        for now change parameters here
        '''
        if bmi is not None:
            self.bmi = bmi
            smooth_taps = 60 #int(2/200e-3)
            self.bmi_pos = np.zeros((smooth_taps, 2))
            self.log.info('initiate the BMI decoder and playground jov connection')
            @self.bmi.binner.connect
            def on_decode(X):
                # print(self.binner.nbins, self.binner.count_vec.shape, X.shape, np.sum(X))
                with Timer('decoding', verbose=False):
                    if self.bmi.dec.name == 'NaiveBayes':
                        X = np.sum(X, axis=0)
                    self.bmi_pos = np.vstack((self.bmi_pos[1:, :], self.bmi.dec.predict(X)))
                    y = np.mean(self.bmi_pos, axis=0)
                    # !!! In the GUI: select task first. Otherwise jov is not initiated 
                    if self.task_name == 'JUMPER':
                        self.jov.teleport(prefix='console', target_pos=(y[0], y[1], 15))
                    elif self.task_name == 'JEDI':
                        self.jov.teleport(prefix='model', target_pos=(y[0], y[1], 15), target_item='_dcue_001')
                    print('pos:{0}, time:{1:.5f} secs'.format(y, self.bmi.binner.current_time))
                    os.write(self.bmi.dec_result, np.hstack((self.bmi.binner.last_bin, y)))


            self.fet_view_timer = QtCore.QTimer(self)
            self.fet_view_timer.timeout.connect(self.fet_view_update)
            self.prb_view.highlight(self.bmi.fpga.configured_groups)
            # self.prb_view_timer = QtCore.QTimer(self)
            # self.prb_view_timer.timeout.connect(self.prb_view_update)
            self.prb_view_frame = 1
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

        BtnLayout = QGridLayout()
        BtnLayout.addWidget(self.vrBtn,0,1)
        BtnLayout.addWidget(self.fpgaBtn,0,0)
        BtnLayout.addWidget(self.toggle_motion_Btn, 1,0)

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
        ParaLayout.addWidget(self.touch_radius_label,  0,2,1,1)
        ParaLayout.addWidget(self.touch_radius,        0,3,1,1)

        #5. Probe View
        self.prb_view = probe_view()
        self.prb_view.set_data(self.prb, font_size=23)

        #6. Feature View
        N=5000
        self.fet_view0 = scatter_3d_view()
        self.fet_view0.set_data(np.zeros((N,4), dtype=np.float32))
        # self.fet_view1 = scatter_3d_view()
        # self.unit_table = QTableWidget()
        # self.unit_table.setColumnCount(5)
        # self.unit_table.setRowCount(8)

        #7. Navigation view for both viz and interaction 
        self.nav_view = maze_view()
        self._maze_loaded   = False
        self._task_selected = False

        #widget layout
        leftlayout = QVBoxLayout()
        leftlayout.addLayout(DirLayout)
        leftlayout.addLayout(BtnLayout)
        leftlayout.addWidget(self.prb_view.native)
        leftlayout.addWidget(self.fet_view0.native)
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
            self.log.info('initiate Jovian and its socket connection')
            self.jov = Jovian()
            self.jov.log = self.log
            self.jov.cnt.fill_(0)
            self.nav_view.connect(self.jov)  # shared cue_pos, shared tranformation
            self.toggle_motion_Btn.clicked.connect(self.jov.toggle_motion)

            # 2. Init Task
            try:
                self.task = globals()[self.task_name](self.jov)
                self.log.info('task: {}'.format(self.task_name))

                # 3. Task parameter
                self.task.reward_time = self.reward_time.value()/10. 
                self.jov.touch_radius.fill_(self.touch_radius.value())
                self.log.info('task reward time: {}, task touch radius: {}'.format(self.task.reward_time, self.jov.touch_radius))
                self.log.info('Task Ready')
                self._task_selected = True

            except:
                self.jov.pynq.shutdown(2)
                raise
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
        self.log.info('load {} {}'.format(maze_mesh_file, maze_coord_file))

        for file in cue_files:
            _cue_file = os.path.join(folder, file)
            self.nav_view.load_cue(cue_file=_cue_file, cue_name=file.split('.')[0])
            self.log.info('load {}'.format(_cue_file))

        self._maze_loaded = True


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
            self.log.warn('select Task First')


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
            self.nav_view.cue_update()


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
        self.fet_view_timer.start(60)
        # self.prb_view_timer.start(60)


    def fpga_process_stop(self):
        self.log.info('---------------------------------')
        self.log.info('fpga_process_stop')
        self.log.info('---------------------------------')
        self.fpgaBtn.setText('BMI Stream Off')
        self.fpgaBtn.setStyleSheet("background-color: darkgrey")
        self.bmi.stop()
        self.fet_view_timer.stop()
        # self.prb_view_timer.stop()


    # def prb_view_update(self):
    #     scv = self.fpga.spike_count_vector.numpy()
    #     # self.log.info('{}:{}'.format('scv', scv))
    #     self.prb_view.set_scv(scv, 15)
    #     scv = np.append(self.prb_view_frame, scv)
    #     scv.tofile('./scv.bin')
    #     self.fpga.spike_count_vector[:] = 0
    #     self.prb_view_frame += 1


    def fet_view_update(self):
        N = 5000
        try:
            fet = np.fromfile('./fet.bin', dtype=np.int32)
            if fet.shape[0] > 0:
                fet = fet.reshape(-1, 7)
                fet_info = fet[:,:2]
                fet_val = fet[:,2:6]
                labels  = fet[:, -1]
                # get idx of fet from current selected group
                idx = np.where(fet_info[:,1]==self.current_group)[0]
                if len(idx)>N:
                    idx = idx[-N:]

                if self.current_group in self.bmi.fpga.configured_groups:
                    fet = fet_val[idx, :]/float(2**16)
                    clu = labels[idx]

                # self.log.info('get_fet{}'.format(idx.shape))
                if len(fet)>0:
                    self.fet_view0.stream_in(fet, clu, highlight_no=30)
        except:
            pass