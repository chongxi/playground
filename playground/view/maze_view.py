# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018, Spiketag team. All Rights Reserved.
# -----------------------------------------------------------------------------

"""
Dynamic update the rat navigational trajectory
In a loaded VR Maze
"""

import os
import numpy as np
from vispy import app, gloo, visuals, scene
from vispy.util import keys
from vispy.visuals.transforms import STTransform, MatrixTransform
from PyQt5.QtWidgets import QFileDialog

from ..base import maze
from ..utils import *
from ..view import Maze, Line, Animal, Cue 
from torch import multiprocessing
from spiketag.view.color_scheme import palette
from spiketag.analysis.core import get_hd


neuro_colors = np.array(palette)

class maze_view(scene.SceneCanvas):
    """
        docstring for navigation_view
    """

    def __init__(self, show=False, debug=False):
        scene.SceneCanvas.__init__(self, title='Navigation View', keys=None)
        self.unfreeze()

        self.free_timer = app.Timer() # use it to connect to whatever

        self.is_jovian_connected = False
        ### 1. viewbox, border and camera
        self.view = self.central_widget.add_view(margin=10)
        # self.view.camera = 'panzoom'
        self.view.camera = 'turntable'
        # self.view.camera = 'fly'
        # self.view.camera = 'perspective'
        self.view.border_color = (1,1,1,0.6)

        ### 2. moving trajectory (line) and current position (marker)
        self.grid           = scene.visuals.GridLines(parent=self.view.scene) # scale=(np.pi/6., 1.0), 
        self.pos            = None
        self.trajectory     = Line(parent=self.view.scene)  #, width=1, antialias=False
        self.ray_vector_visible = False
        self.ray_vector  = scene.visuals.Line(parent=self.view.scene)

        self._current_pos = None 
        self._current_hd  = 0.0
        self.marker       = None    # visualize animal position 
        self.arrow        = None    # visualize animal head direction
        self._arrow_len    = 1000.0

        self.SWR_trajectory = scene.visuals.Line(parent=self.view.scene)
        self.animal_color   = 'white'

        self.cues = {}
        self.cues_height = {}
        self._selected_cue = None

        ### 3. replay
        self.replay_current_pos = scene.visuals.Markers(parent=self.view.scene)
        self.replay_current_pos.set_data(np.array([0,0,0]).reshape(-1,3))
        self.replay_current_hd = scene.visuals.Arrow(parent=self.view.scene, width=1.2, color=(0,1,0,1))
        hd = np.array([[0,0],[100,100]])
        self.replay_current_hd.set_data(pos=hd, arrows=hd.reshape((len(hd)//2, 4)))
        self.replay_trajectory  = scene.visuals.Line(parent=self.view.scene, color=(0.7, 0.7, 0.7, 1))
        self.replay_time = 0.0
        self.replay_coord      = 'jovian'
        self.replay_timer = app.Timer()
        self.replay_timer.connect(self.on_replay)
        self.replay_speed = 1

        ### Timer
        self._timer = app.Timer(0.1)
        self._timer.connect(self.on_timer)
        # self._timer.start(0.8)
        self.global_i = 0

        ### random walk timer, update each 100 ms
        self._random_walk_timer = app.Timer()
        self._random_walk_timer.connect(self.on_random_walk_timer)
        self._random_walk_step = 0

        ### 4. background and fields
        self.image_background = scene.visuals.Image(parent=self.view.scene, method='subdivide')
        self.image_background.transform = STTransform()
        self.image = scene.visuals.Image(parent=self.view.scene, method='subdivide', cmap='hot', clim=[0.025, 0.3])
        self.image.transform = STTransform()

        ### 5. distance check
        self.goal_distances = f'Distances: 0, 0, 0'
        self.goal_distances_text = scene.visuals.Text(parent=self.scene)
        self.goal_distances_text.text = self.goal_distances
        self.goal_distances_text.pos = np.array([130, 30])
        self.goal_distances_text.color = (1, 1, 1, 0.7)
        self.goal_distances_text.font_size = 10
        self._goal_distance_timer = app.Timer()
        self._goal_distance_timer.connect(self.check_goal_distance)
        self._goal_distance_timer.start(0.1)

        # self.set_range()
        # self.freeze()
        ### first person view
        self.fpv = False

    def check_goal_distance(self, event):
        if self.current_pos is not None:
            try:
                animal_pos = self.maze_2_real_pos(self.current_pos[:2])
                cue0_pos = self.cues['_dcue_000'].pos[:2]
                cue1_pos = self.cues['_dcue_001'].pos[:2]
                self.goal_distances = (np.linalg.norm(animal_pos-cue0_pos),
                                    np.linalg.norm(animal_pos-cue1_pos),
                                    np.linalg.norm(cue0_pos-cue1_pos))
                self.goal_distances_text.text = f'Distances: {self.goal_distances[0]:.1f}, {self.goal_distances[1]:.1f}, {self.goal_distances[2]:.1f}'
            except:
                pass

    def random_walk(self, n_steps=10000, init_pos=[0, 0], max_speed=10, dt=0.1, smooth_factor=5):
        if self.current_pos is not None:
            init_pos = self.maze_2_real_pos(self.current_pos)
        self.random_walk_pos, self.random_walk_hd, v = randomwalk2D(n_steps, init_pos,
                                                                    x_range=self.x_range_real,
                                                                    y_range=self.y_range_real,
                                                                    max_speed=max_speed,
                                                                    smooth_factor=smooth_factor,
                                                                    dt=dt)
        self._random_walk_timer.start(dt)

    def on_random_walk_timer(self, event):
        if self._random_walk_step >= len(self.random_walk_pos):
            self._random_walk_timer.stop()
            self._random_walk_step = 0
        else:
            self.current_pos = self.real_2_maze_pos(self.random_walk_pos[self._random_walk_step])
            self.current_hd  = self.random_walk_hd[self._random_walk_step]%360 + 90
            self._random_walk_step += 1

    def load_all(self):
        base_folder = os.path.dirname(maze.__file__)
        folder = QFileDialog.getExistingDirectory(None,'', base_folder, QFileDialog.ShowDirsOnly)

        maze_files = [_ for _ in os.listdir(folder) if 'maze' in _]
        cue_files  = [_ for _ in os.listdir(folder) if 'cue'  in _]
        print(maze_files)
        print(cue_files)
        for file in maze_files:
            if file.endswith(".obj"):
                maze_mesh_file = os.path.join(folder, file)
            elif file.endswith(".coords"):
                maze_coord_file = os.path.join(folder, file)
        self.load_maze(maze_file = maze_mesh_file, 
                       maze_coord_file = maze_coord_file) 
        self.load_animal()

        for file in cue_files:
            _cue_file = os.path.join(folder, file)
            self.load_cue(cue_file=_cue_file, cue_name=file.split('.')[0])


    def load_maze(self, maze_file, mirror=True, maze_coord_file=None):
        self.maze = Maze(maze_file, maze_coord_file) #color='gray'

        self.scale_factor = 100
        self.origin    = -np.array(self.maze.coord['Origin']).astype(np.float32) * self.scale_factor
        self.origin_hd = np.arctan2(-self.origin[1], self.origin[0])/np.pi*180
        self.border  = np.array(self.maze.coord['border']).astype(np.float32)
        self.x_range_real = self.border[[0,2]]
        self.y_range_real = self.border[[1,3]]
        self.x_range = (self.origin[0]+self.border[0]*self.scale_factor, self.origin[0]+self.border[2]*self.scale_factor)
        self.y_range = (self.origin[1]+self.border[1]*self.scale_factor, self.origin[1]+self.border[3]*self.scale_factor)
        self._arrow_len = (self.x_range[1]-self.x_range[0])/10
        # self.marker.move(self.origin[:2])
        # self.current_pos = self.origin[:2]

        ### MatrixTransform perform Affine Transform
        transform = MatrixTransform()
        # transform.rotate(angle=90, axis=(1, 0, 0))  # rotate around x-axis for 90, the maze lay down
        if mirror:
            self.mirror = True
            transform.matrix[:,2] = - transform.matrix[:,2]  # reflection matrix, mirror image on x-y plane
        transform.scale(scale=4*[self.scale_factor]) # scale at all 4 dim for scale_factor
        transform.translate(pos=self.origin) # translate to origin

        self.maze.transform = transform 
        self.view.add(self.maze)
        self.set_range()
        print('Origin:', self.origin)
        print('border:', self.border)


    def load_animal(self):
        self.marker = Animal(parent=self.view.scene, radius=200, color=(1,0,1,0.8))
        self.marker.transform = STTransform()
        self.marker.origin = self.origin
        self.arrow = scene.visuals.Arrow(parent=self.view.scene, width=1.2, color=(0,1,0,1))
        hd = np.array([[0,0],[100,100]])
        self.arrow.set_data(pos=hd, arrows=hd.reshape((len(hd)//2, 4)))

        @self.marker.connect
        def on_move(target_pos):
            if hasattr(self, 'jov'):
                self.jov.teleport(prefix='console', target_pos=target_pos)
            else:
                self.current_pos = self._to_jovian_coord(target_pos).astype(np.float32)


    def load_cue(self, cue_file, cue_name=None):
        _cue = Cue(cue_file) 
        _cue.name = cue_name
        self.cues[cue_name] = _cue
        self.cues[cue_name].center = self.maze.coord[cue_name]
        self.cues[cue_name].origin = self.origin
        self.cues[cue_name].transform = STTransform()
        self.cues[cue_name].scale(100)
        self.cues[cue_name].pos = [0, 0, self.cues[cue_name].center[-1]]
        self.view.add(self.cues[cue_name])

        self.cues_height[cue_name] = self.cues[cue_name].center[-1]  # jovian will use this papameter

        @_cue.connect
        def on_move(target_item, target_pos):
            if hasattr(self, 'jov'):
                self.jov.teleport(prefix='model', target_pos=target_pos, target_item=target_item)
            else:
                _cue_default_offset = self.cues[target_item]._xy_center*self.cues[target_item]._scale_factor
                self.cues[target_item]._transform.translate = self._to_jovian_coord(target_pos).astype(np.float32) - _cue_default_offset


    def cue_update(self):
        with Timer('cue_update', verbose = False):
            for cue_name, cue_pos in self.shared_cue_dict.items():
                # print cue_name, cue_pos
                _cue = self.cues[cue_name]
                # [1,1,-1] because of mirror image of projection
                _cue._transform.translate = np.array([1,1,-1])*cue_pos - _cue._xy_center*_cue._scale_factor


    def set_file(self, file):
        self.unfreeze()
        self._file = file
        if file[-3:] == 'npy':
            pos = np.load(file).astype(np.float32)
            self.pos = pos


    def show_trajectory(self, pos):
        if self.replay_coord == 'jovian':
            pos = self._to_jovian_coord(pos).astype(np.float32)
        self.replay_trajectory.set_data(pos)
        self.replay_current_pos.set_data(pos[0].reshape(-1,3))
            
            
    def load_replay_file(self, file_name, var='pos', show=True):
        # pos = np.load(file_name)[var]
        t = np.load(file_name)['time']
        pos = np.load(file_name)['pos']
        t, pos = interp_pos(t, pos)

        if self.replay_coord == 'jovian':
            pos = self._to_jovian_coord(pos).astype(np.float32)
        
        self.replay_t = t
        self.replay_pos = pos
        if show:
            self.replay_trajectory.set_data(self.replay_pos)
            self.replay_current_pos.set_data(self.replay_pos[0].reshape(-1,3))

    def load_spktime_file(self, file_name, var='spk_time'):
        self.replay_spk_t = np.load(file_name, allow_pickle=True, encoding='bytes')[var].item()        

    def load_neurons(self, file_name=None, var='spk_time', spk_time=None):
        if file_name is None:
            self.replay_spk_t = spk_time
        else:
            self.load_spktime_file(file_name, var)
        self.neurons = {}
        self.neuron_firing_pos = {}
        self.neuron_color = {}
        for i in self.replay_spk_t.keys():
            self.neurons[i] = scene.visuals.Markers(parent=self.view.scene)
            self.neuron_firing_pos[i] = np.array([])
            self.neuron_color[i] = np.array([])
        # print self.replay_spk_t
    
    def on_replay(self, event):
        ### update the replay_time (forward with replay_speed)
        last_time = self.replay_time
        self.replay_time += self.replay_speed * 0.01

        ### update the trajectory 
        idx = np.where(self.replay_t < self.replay_time)[0]
        self.replay_trajectory.set_data(self.replay_pos[idx])

        ### update the last position (current position)
        current_pos = self.replay_pos[idx[-1]].reshape(-1,3)
        self.replay_current_pos.set_data(current_pos, face_color=(1,1,1,0.5))

        ### update head direction (hd_window controls the #samples to get_hd)
        hd_window = 30
        hd_drawlen = 1000
        if len(idx)>hd_window:
            hd, speed = get_hd(trajectory=self.replay_pos[idx][-hd_window:], speed_threshold=10, offset_hd=0)
        else:
            hd, speed = get_hd(trajectory=self.replay_pos[idx], speed_threshold=10, offset_hd=0)
        dx, dy = np.sin(hd/360*np.pi*2), np.cos(hd/360*np.pi*2) 
        arrow = np.vstack(( current_pos.ravel()[:2], current_pos.ravel()[:2] + hd_drawlen * np.array([dx,dy])) ) 
        self.replay_current_hd.set_data(arrow)

        ## neuron_id is the list of neurons to replay
        # neuron_id = self.neuron_id
        # print("replay neuron ids={}".format(self.neuron_id))
        for neuron_id in self.neuron_id:
            neuron_firing = np.where(np.logical_and(self.replay_spk_t[neuron_id] > last_time,
                                                    self.replay_spk_t[neuron_id] < self.replay_time))[0]
            if neuron_firing.shape[0] > 0:
                for i in range(neuron_firing.shape[0]):
                    # neuron_sound(frequency=250.0, duration=0.011)
                    pass
                self.neuron_firing_pos[neuron_id] = np.append(self.neuron_firing_pos[neuron_id], 
                                                              current_pos).reshape(-1, 3)
                intensity = neuron_firing.shape[0]*0.2
                intensity = intensity if intensity<=1 else 1.0
                # intensity = 1
                self.neuron_color[neuron_id] = np.append(self.neuron_color[neuron_id], 
                                                         np.array([neuro_colors[neuron_id][0], 
                                                                   neuro_colors[neuron_id][1], 
                                                                   neuro_colors[neuron_id][2] , 
                                                                   intensity])).reshape(-1,4)
                self.neurons[neuron_id].set_data(self.neuron_firing_pos[neuron_id], 
                                                 face_color=self.neuron_color[neuron_id], 
                                                 size=20)


    @property
    def current_pos(self):
        return self._current_pos


    @current_pos.setter
    def current_pos(self, pos_in):
        '''
        ! pos_in must be np array with shape (2,): (vr_x, vr_y)
        '''
        pos_in = np.array(pos_in).astype(np.float32)
        self._current_pos = pos_in
        self.marker.transform.translate = self._current_pos
        self.stream_in_pos(pos_in)

        if self.fpv is True:
            self.view.camera._center = (pos_in[0], pos_in[1], 100)
            self.view.camera.azimuth = self.current_hd # look ahead
            self.view.camera.view_changed()

    @property
    def real_pos(self):
        return (self.pos-self.origin[:2])/self.scale_factor

    def maze_2_real_pos(self, pos):
        return (pos-self.origin[:2])/self.scale_factor

    def real_2_maze_pos(self, pos):
        return (pos*self.scale_factor)+self.origin[:2]

    @property
    def current_hd(self):
        return self._current_hd

    @current_hd.setter
    def current_hd(self, hd_in):
        self._current_hd = hd_in # absolute value from rotation encoder
        try:
            _current_hd_calibrated = hd_in - 90 # point ahead (0 towards the upper board, 90 towards the right board)
            self.arrow_delta = np.array([np.cos(_current_hd_calibrated/360*np.pi*2), 
                                         np.sin(_current_hd_calibrated/360*np.pi*2)]).ravel()
            arrow = np.vstack(( self.current_pos[:2], 
                                self.current_pos[:2] + self._arrow_len * self.arrow_delta ))
            assert(arrow.shape==(2,2)) # first row is current_pos, arrow_delta point into the head direction
            self.arrow.set_data(arrow)
        except:
            pass

    @property
    def arrow_len(self):
        return self._arrow_len

    @arrow_len.setter
    def arrow_len(self, arrow_len_in):
        self._arrow_len = arrow_len_in
        self.current_hd = self._current_hd

    @property
    def posterior(self):
        return self._posterior

    @posterior.setter
    def posterior(self, posterior):
        self._posterior = posterior
        self.image.set_data(posterior)
        if self.mirror:
            self.image.transform.translate = np.array([self.x_range[0], self.y_range[0], 100])
        else:
            self.image.transform.translate = np.array([self.x_range[0], self.y_range[0], -100])
        ## posterior.shape[1] is the #bins in the x_range
        ## posterior.shape[0] is the #bins in the y_range
        self.image.transform.scale = ((self.x_range[1] - self.x_range[0])/posterior.shape[1], 
                                      (self.y_range[1] - self.y_range[0])/posterior.shape[0])
        self.image.update()


    def stream_in_pos(self, pos_in):
        '''
        ! pos_in must be np array with shape (2,) because the way we write LineVisual
          It only affects visualzation but not computation involved in task
        '''
        # print pos_in
        pos_in = pos_in[:2]
        if self.pos is None:
            self.pos = pos_in.reshape(-1,2)
        else:
            self.pos = np.vstack((self.pos, pos_in.reshape(-1,2))) 
        N = self.pos.shape[0]
        # print self.pos.shape
        # if N > 50000:
        #     self.pos = np.delete(self.pos, 0)
        dynamic_color = np.ones((N, 4), dtype=np.float32) 
        dynamic_color[:, 0] = np.linspace(0, 1, N)
        dynamic_color[:, 1] = dynamic_color[::-1, 0]
        self.trajectory.set_data(pos=self.pos, color=dynamic_color)


    def set_data(self, pos):
        '''
            Set the trajactory with marker indicate current position
        '''
        self.unfreeze()
        self.pos = pos
        color = np.ones((self.pos.shape[0], 4)) * np.array([1,0,1,0.5])
        self.trajectory.set_data(self.pos, color=color)


    def on_timer(self, event):
            # try:
        if self._selected_cue is not None:
            self.cues[self._selected_cue].vibrate(5)
            # except:
                # self._timer.stop()


    def set_range(self):
        self.view.camera.set_range(x=self.x_range,y=self.y_range, margin=.1)
        self.view.camera.elevation=180+45


    def imap(self, mouse_pos):
        ''' Convert mouse click into Jovian coordination
            mouse_pos is 4 element array that reflects user mouse click position
        '''
        with Timer('imap', verbose=False):
            # Return the transform that maps from the coordinate system of grid to the local coordinate system of node (view).
            # The below three transform are the same
            # tr = self.view.camera._scene_transform
            # tr = self.grid.node_transform(self.view)
            tr = self.view.scene.transform
            rayPoint = tr.imap(mouse_pos)[:3]
            rayDirection = self.view.camera.transform.matrix[2][:3]
            if self.ray_vector_visible:
                u = generate_line(rayPoint, rayDirection)
                self.ray_vector.set_data(pos=u, width=1, color='green')

            planePoint = (0,0,0) # or self.origin
            planeDirection = (0,0,1) # project to xyplane where z=0: (0,0,1) has null space `z=0`
            jovian_pos = line_plane_intersection(rayDirection, rayPoint, planeDirection, planePoint)
            maze_pos = self._to_maze_coord(jovian_pos)
            return maze_pos


    def _to_maze_coord(self, pos):
        '''transform back to maze coord (0,0,0)
        '''
        if pos.ndim == 2:
            if pos.shape[1] == 2:
                pos = np.hstack((pos, np.zeros((pos.shape[0],1))))
        return (pos-self.origin)/self.scale_factor

    def _to_jovian_coord(self, pos):
        '''transform to mouseover coord at origin
        '''
        if pos.ndim == 2:
            if pos.shape[1] == 2:
                pos = np.hstack((pos, np.zeros((pos.shape[0],1))))
        return (pos*self.scale_factor)+self.origin


    def connect(self, jov):
        '''--------------------------------
            connect maze and jovian: 
            1. shared_cue_dict.  := {cue_name: cue_pos}
            2. shared_cue_height := {cue_name: cue_height}
            3. coord transformations
           --------------------------------
        '''        
        self.jov = jov
        mgr = multiprocessing.Manager()
        self.shared_cue_dict = mgr.dict()
        self.jov.set_trigger(self.shared_cue_dict)
        self.jov.shared_cue_height = self.cues_height
        self.jov._to_maze_coord = self._to_maze_coord
        self.jov._to_jovian_coord = self._to_jovian_coord
        self.is_jovian_connected = True
        self.jov.maze_origin = self.origin
        self.jov.maze_scale = self.scale_factor
        self.jov.log.info('jov acquire maze_origin as: %s' % str(self.jov.maze_origin))
        self.jov.log.info('jov acquire maze_scale as: %s' % str(self.jov.maze_scale))

        # @self.jov.connect
        # def on_cue(cue_id, func, args):
        #     '''jov control the cue, cue_id is 0,1,2,3
        #        func is the cue fucntion name, execute with args
        #     '''
        #     getattr(self.cues[self.cues.keys()[cue_id]], func)(args)
            # print(f)
            # f(args)
            # self.cues[target_item].pos = target_pos


    '''--------------------------------
        Below is the interaction code
       --------------------------------
    '''

    def on_key_press(self, e):
        '''
            Control: control + mouse to bypass built-in mouse reaction
            r:       reset the camera
        '''
        if keys.CONTROL in e.modifiers:
            self.view.events.mouse_wheel.disconnect(self.view.camera
                    .viewbox_mouse_event)
        elif e.text == ' ':
            if self.replay_timer.running:
                self.replay_timer.stop()
            else:
                self.replay_timer.start(0.01)
        elif e.text == 's':
            if self.free_timer.running:
                self.free_timer.stop()
            else:
                self.free_timer.start(0.01)
        elif e.text == 'r':
            self.set_range() 
        elif e.text == 'h':
            self.trajectory.visible = not self.trajectory.visible
        elif e.text == 'c':
            self.pos = None
            self.trajectory.update()
        elif e.text == '0':
            self._selected_cue = None
        elif e.text == '1':
            self._selected_cue = list(self.cues.keys())[0]
            # print('{} is selected'.format(self._selected_cue))
        elif e.text == '2':
            self._selected_cue = list(self.cues.keys())[1]
            # print('{} is selected'.format(self._selected_cue))
        elif e.text == '.':
            self.view.camera = 'fly'
            self.set_range()
        elif e.text == ',':
            self.view.camera = 'turntable'
            self.set_range()
        elif e.text == 'o':
            self.load_all()       
        elif e.text == 'q':
            self.view.camera.elevation = -90
            self.view.camera.azimuth = 0
        elif e.text == 'f':
            self.image.visible = not self.image.visible
        elif e.text == 'p':
            self.fpv = not self.fpv  # this will visualize the current_hd as first person view

    def on_mouse_release(self, e):
        if self.marker is not None:
            with Timer('click',verbose=False):
                modifiers = e.modifiers
                if keys.CONTROL in e.modifiers and e.button == 1:
                    self.marker.pos = self.imap(e.pos)
                if keys.CONTROL in e.modifiers and e.button == 2:
                    with Timer('cue moving', verbose=False):
                        target_maze_pos = self.imap(e.pos)
                        # print(target_maze_pos)
                        self.cues[self._selected_cue].pos = target_maze_pos
                        

    def run(self):
        self.show()
        self.app.run()
