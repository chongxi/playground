#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 23:54:33 2018

@author: laic
"""

from playground.view import maze_view

if __name__ == '__main__':
    nav_view = maze_view()
    nav_view.load_maze('../playground/base/maze/obj/maze_2d.obj', '../playground/base/maze/obj/maze_2d.coords')
    # nav_view.load_cue(cue_file='../playground/base/maze/obj/constraint_cue.obj', cue_name='_dcue_001')
    # nav_view.load_cue(cue_file='../playground/base/maze/obj/goal_cue.obj', cue_name='_dcue_000')
    nav_view.load_replay_file('./task_replay.npz', show=False)
    # nav_view.load_neurons('./timing.npz', 'timing')
    nav_view.replay_speed = 5
    nav_view.replay_time = 0
    nav_view.neuron_id = range(1,9)
    # nav_view.replay_timer.start(0.01)
    nav_view.run()