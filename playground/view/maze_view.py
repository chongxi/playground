# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018, Spiketag team. All Rights Reserved.
# -----------------------------------------------------------------------------

"""
Dynamic update the rat navigational trajectory
In a loaded VR Maze
"""

import numpy as np
from vispy import app, gloo, visuals, scene
from vispy.util import keys
from vispy.visuals.transforms import STTransform, MatrixTransform

from ..utils import *
from ..view import Maze, Line, Animal, Cue 
from torch import multiprocessing
from pysine import sine



class maze_view(scene.SceneCanvas):
    """
        docstring for navigation_view
    """

    def __init__(self, show=False, debug=False):
        scene.SceneCanvas.__init__(self, title='Navigation View', keys=None)
        self.unfreeze()

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
        self.marker       = None

        self.SWR_trajectory = scene.visuals.Line(parent=self.view.scene)
        self.animal_color   = 'white'
        
        ### 4. replay
        self.replay_current_pos = scene.visuals.Markers(parent=self.view.scene)
        self.replay_current_pos.set_data(np.array([0,0,0]).reshape(-1,3))
        self.replay_trajectory  = scene.visuals.Line(parent=self.view.scene)
        self.replay_time = 0.0
        self.replay_coord      = 'jovian'
        self.replay_timer = app.Timer()
        self.replay_timer.connect(self.on_replay)
        self.replay_speed = 1

        ### 4. cue objects
        self.cues = {}
        self.cues_height = {}
        self._selected_cue = None

        ### Timer
        self._timer = app.Timer(0.1)
        self._timer.connect(self.on_timer)
        # self._timer.start(0.8)
        self.global_i = 0

        # self.set_range()
        # self.freeze()


    def load_maze(self, maze_file, maze_coord_file=None, border=[-100,-100,100,100]):
        self.maze = Maze(maze_file, maze_coord_file) #color='gray'

        self.scale_factor = 100
        self.origin  = -np.array(self.maze.coord['Origin']).astype(np.float32) * self.scale_factor
        self.border  = border
        self.x_range = (self.origin[0]+self.border[0]*self.scale_factor, self.origin[0]+self.border[2]*self.scale_factor)
        self.y_range = (self.origin[1]+self.border[1]*self.scale_factor, self.origin[1]+self.border[3]*self.scale_factor)
        # self.marker.move(self.origin[:2])
        # self.current_pos = self.origin[:2]

        ### MatrixTransform perform Affine Transform
        transform = MatrixTransform()
        transform.rotate(angle=90, axis=(1, 0, 0))  # rotate around x-axis for 90, the maze lay down
        transform.matrix[:,2] = - transform.matrix[:,2]  # reflection matrix, mirror image on x-y plane
        transform.scale(scale=4*[self.scale_factor]) # scale at all 4 dim for scale_factor
        transform.translate(pos=self.origin) # translate to origin

        self.maze.transform = transform 
        self.view.add(self.maze)
        self.set_range()
        print('Origin:', self.origin)


    def load_animal(self):
        self.marker = Animal(parent=self.view.scene, radius=200, color=(1,0,1,0.8))
        self.marker.transform = STTransform()
        self.marker.origin = self.origin

        @self.marker.connect
        def on_move(target_pos):
            self.jov.teleport(prefix='console', target_pos=target_pos)


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
            self.jov.teleport(prefix='model', target_pos=target_pos, target_item=target_item)


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

    def load_neurons(self, file_name, var='spk_time'):
        self.replay_spk_t = np.load(file_name)[var].item()
        self.neurons = scene.visuals.Markers(parent=self.view.scene)
        self.neuron_firing_pos = {}
        for i in self.replay_spk_t.keys():
            self.neuron_firing_pos[i] = np.array([])
        # print self.replay_spk_t

    
    def on_replay(self, event):
        last_time = self.replay_time
        self.replay_time += self.replay_speed * 0.01
        idx = np.where(self.replay_t < self.replay_time)[0]
        self.replay_trajectory.set_data(self.replay_pos[idx])
        current_pos = self.replay_pos[idx[-1]].reshape(-1,3)
        self.replay_current_pos.set_data(current_pos)

        neuron_id = self.neuron_id
        neuron_firing = np.where(np.logical_and(self.replay_spk_t[neuron_id] > last_time,
                                                self.replay_spk_t[neuron_id] < self.replay_time))[0]
        if neuron_firing.shape[0] > 0:
            for i in range(neuron_firing.shape[0]):
                sine(frequency=250.0, duration=0.011)
            self.neuron_firing_pos[neuron_id] = np.append(self.neuron_firing_pos[neuron_id], 
                                                          current_pos).reshape(-1, 3)
            self.neurons.set_data(self.neuron_firing_pos[neuron_id], face_color=(1,1,0,0.5), size=8)




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
            self._selected_cue = self.cues.keys()[0]
            # print('{} is selected'.format(self._selected_cue))
        elif e.text == '2':
            self._selected_cue = self.cues.keys()[1]
            # print('{} is selected'.format(self._selected_cue))
        elif e.text == '.':
            self.view.camera = 'fly'
            self.set_range()
        elif e.text == ',':
            self.view.camera = 'turntable'
            self.set_range()

    def on_mouse_release(self, e):
        if self.marker is not None:
            with Timer('click',verbose=False):
                modifiers = e.modifiers
                if keys.CONTROL in e.modifiers and e.button == 1:
                    self.marker.pos = self.imap(e.pos)
                if keys.CONTROL in e.modifiers and e.button == 2:
                    with Timer('cue moving', verbose=False):
                        target_maze_pos = self.imap(e.pos)
                        self.cues[self._selected_cue].pos = target_maze_pos
                        

    def run(self):
        self.show()
        self.app.run()
