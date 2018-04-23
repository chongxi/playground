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


class maze_view(scene.SceneCanvas):
    """
        docstring for navigation_view
    """

    def __init__(self, show=False, debug=False):
        scene.SceneCanvas.__init__(self, title='Navigation View', keys=None)
        self.unfreeze()

        self.is_sock_cmd_connected = False
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
        # self.marker    = scene.visuals.Markers(parent=self.view.scene)
        self.marker = Animal(parent=self.view.scene, radius=200, color=(1,0,1,0.8))
        self.marker.transform = STTransform()

        self.SWR_trajectory = scene.visuals.Line(parent=self.view.scene)
        self.animal_color   = 'white'


        ### 4. cue objects
        self.cues = {}
        self._selected_cue = 0

        ### Timer
        self._timer = app.Timer(0.1)
        self._timer.connect(self.on_timer)
        self.global_i = 0

        # self.set_range()
        # self.freeze()


    def load_maze(self, maze_file, maze_coord_file=None, border=[-100,-100,100,100]):
        self.maze = Maze(maze_file, maze_coord_file) #color='gray'

        self.scale_factor = 100
        self.origin  = -np.array(self.maze.coord['Origin']) * self.scale_factor
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
        print(self.maze.coord['Origin'])

    def load_cue(self, cue_file, cue_name=None):
        self.cues[cue_name] = Cue(cue_file) 
        self.cues[cue_name].center = self.maze.coord[cue_name]
        self.cues[cue_name].transform = STTransform()
        self.cues[cue_name].scale(100)
        self.cues[cue_name].move(self.origin)
        self.view.add(self.cues[cue_name])


    def set_file(self, file):
        self.unfreeze()
        self._file = file
        if file[-3:] == 'npy':
            pos = np.load(file).astype(np.float32)
            self.pos = pos
            

    @property
    def current_pos(self):
        return self._current_pos


    @current_pos.setter
    def current_pos(self, pos_in):
        '''
        ! pos_in must be np array with shape (2,): (vr_x, vr_y)
        '''
        pos_in = pos_in.astype(np.float32)
        self._current_pos = pos_in
        self.marker.transform.translate = self._current_pos
        self.stream_in_pos(pos_in)


    def stream_in_pos(self, pos_in):
        '''
        ! pos_in must be np array with shape (2,)
        '''
        # print pos_in
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
        self.global_i += 100
        with Timer('rendering takes', verbose = False):
            try:
                dynamic_pos   = self.pos[:self.global_i]
                N = dynamic_pos.shape[0]
                print N
                dynamic_color = np.ones((N, 4), dtype=np.float32)
                dynamic_color[:, 0] = np.linspace(0, 1, N)
                dynamic_color[:, 1] = dynamic_color[::-1, 0]
                self.trajectory.set_data(pos=dynamic_pos, color=dynamic_color)
                self.marker.transform.translate = dynamic_pos[-1]
                # self.marker.set_data(pos=dynamic_pos[-1].reshape(-1,2), face_color=self.animal_color)

            except:
                self._timer.stop()

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
            maze_pos = line_plane_intersection(rayDirection, rayPoint, planeDirection, planePoint)
            return maze_pos


    '''--------------------------------
        user issued socket commands 
       --------------------------------
    '''

    def connect(self, sock_cmd):
        '''connect to the socket command output
        '''
        self.sock_cmd = sock_cmd
        self.is_sock_cmd_connected = True

    def teleport(self, prefix, target_item, target_pos):
        '''
           Core function: This is the only function that send `events` back to Jovian from interaction 
        '''
        if self.is_sock_cmd_connected:
            x, y, _ = (target_pos-self.origin)/self.scale_factor # the maze coordination
            v = 0 # control how animal move update scene, should be always 0
            if prefix == 'console':  # teleport animal
                z = 5 # this should always be 5 based on Jovian constraint
                cmd = "{}.teleport({},{},{},{})\n".format(prefix, x,y,z,v)
                self.sock_cmd.send(cmd)
            elif prefix == 'model':  # move cue
                z = self.cues[target_item].center[-1]
                print 'z value', z
                cmd = "{}.move('{}',{},{},{})\n".format(prefix, target_item, x, y, z)
                self.sock_cmd.send(cmd)
        else:
            pass
        # print('from trajectory_view:',cmd)


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
        elif e.text == '1':
            self._selected_cue = 0
            print(self.cues.keys()[self._selected_cue])
        elif e.text == '2':
            self._selected_cue = 1
            print(self.cues.keys()[self._selected_cue])
        elif e.text == '.':
            self.view.camera = 'fly'
            self.set_range()
        elif e.text == ',':
            self.view.camera = 'turntable'
            self.set_range()

    def on_mouse_release(self, e):
        with Timer('click',verbose=False):
            modifiers = e.modifiers
            if keys.CONTROL in e.modifiers and e.button == 1:
                target_item = 'animal'
                target_pos  = self.imap(e.pos)
                self.teleport(prefix='console', target_item=target_item, target_pos=target_pos)
            if keys.CONTROL in e.modifiers and e.button == 2:
                with Timer('cue moving', verbose=False):
                    target_item = self.cues.keys()[self._selected_cue]
                    target_pos  = self.imap(e.pos)
                    self.cues[target_item].move(target_pos)
                    self.teleport(prefix='model', target_item=target_item, target_pos=target_pos)


    def run(self):
        self.show()
        self.app.run()
