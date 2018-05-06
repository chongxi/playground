import numpy as np
from ..jovian import Jovian
from ...utils import EventEmitter
import time
import torch
from itertools import chain
from collections import deque, namedtuple
from abc import abstractmethod


def _cue_generate_2d_maze(*args):
    '''generate a cue position that is not going to be near to the _pos in args
       low, high are border condition for cue generation
       function should return < 100us
    '''
    radius = 50 

    while True:
        break_condition = True
        pos = np.random.randint(low=-85, high=85, size=(2,))
        for _pos in args:
            if np.linalg.norm(pos - np.array(_pos)) < radius:
               break_condition = False 
        if break_condition:
            break
    return pos


class TrouchEvent(EventEmitter):
    """docstring for Event"""
    def __init__(self, _id, _coord):
        super(TrouchEvent, self).__init__()
        self.type = 'touch'
        self.what = _id
        self.where = _coord

        
class Task(object):
    '''Every task has the same jov to report event
       Every task has different fsm is defined in child task class
    ''' 
    def __init__(self, fsm, jov):
        # jovian first
        self.jov = jov
        self.log = jov.log
        self.fsm = fsm 
        self.transition_enable = namedtuple("transition_enable", ['behave', 'env', 'ephys'], verbose=False)
        self.transition_enable.behave, self.transition_enable.ephys = True, False
        self.ani = EventEmitter()
        self.animation = {}  # {"_dcue_000": deque([ (4, parachute), (60, vibrate) ]), "_dcue_111": deque([ (2, animation111) ])}

        # initial reset only after jov process start 
        @self.jov.connect
        def on_start():
            self.reset()  

        # fsm will use jov real-time event to update
        @self.jov.connect
        def on_touch(args):
            self.touch_id, self.coord = args
            self.event = TrouchEvent(self.touch_id, self.coord)
            self.on_event(self.event) 

        # use jov frame update to do the animation
        @self.jov.connect
        def on_frame():
            for _cue_name, _action_queue in self.animation.items():
                try: 
                    _pace, _animation = _action_queue[0]
                    if self.jov.cnt % _pace == 0:
                        try:
                            _animation.next()
                        except StopIteration:
                            __pace__, __animation__ = _action_queue.popleft()
                            self.ani.emit('animation_finish', animation_name=__animation__.__name__)
                except:
                    pass

        @self.ani.connect
        def on_animation_finish(animation_name):
            self.log.info('{} finished'.format(animation_name))


    @abstractmethod
    def reset(self):
        self.transition_enable.behave = True

    @property
    def current_pos(self):
        '''shared memory of current pos of animal
        '''
        return self.jov.current_pos.numpy()

    def __call__(self, fsm):
        self.fsm = fsm
        self.reset()

    def on_event(self, event):
        if event.type == 'touch':
            if self.transition_enable.behave:
                # try:
                self.log.info('state: {}, {}: {}@{}'.format(self.state, event.type, event.what, event.where))
                next_state, func, args = self.fsm[self.state][event.what]
                func(args)
                self.state = next_state
                self.log.info('state: {}'.format(self.state))
                # except:
                    # self.log.warn('Your Finite State Machine is Incomplete or Wrong')

        elif event.type == 'ephys':
            #TODO: Add ephys API
            pass

    #------------------------------------------------------------------------------
    # animation effect
    #------------------------------------------------------------------------------

    def parachute(self, cue_name, pos):
        ''' usage:
            self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)) ])
        '''
        for z in range(60,-1,-2):
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  z], target_item=cue_name)
            yield

    def bury(self, cue_name):
        self.transition_enable.behave = False
        for z in range(0, -100, -8):
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  z], target_item=cue_name)
            yield

    def vibrate(self, cue_name):
        ''' usage:
            self.animation['_dcue_001'] = deque([ (3, self.parachute('_dcue_001', self._coord_guide)), (30, self.vibrate('_dcue_001')) ])
        '''
        for z in range(1000):
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  5], target_item=cue_name)
            yield
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  0], target_item=cue_name)
            yield            

    def wander(self, cue_name, direction='x'):
        for i in range(10000):
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            if direction=='x':
                _x = pos[0]
                while _x>=-95:
                    _x -= 5
                    self.jov.teleport(prefix='model', target_pos=[_x,  pos[1],  0], target_item=cue_name)
                    yield
                while _x<=95:
                    _x += 5
                    self.jov.teleport(prefix='model', target_pos=[_x,  pos[1],  0], target_item=cue_name)
                    yield

    def escape(self, cue_name, speed=5):
        while True:
            animal_pos = self.jov._to_maze_coord(self.current_pos)
            cue_pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            yield
            new_animal_pos = self.jov._to_maze_coord(self.current_pos) 
            new_cue_pos = new_animal_pos - animal_pos + cue_pos
            self.jov.teleport(prefix='model', target_pos=[new_cue_pos[0],  new_cue_pos[1],  0], target_item=cue_name)
            yield


    def trajectory_teleport(self, trajectory):
        ''' usage:
            trajectory = np.arange(-99,100).repeat(2).reshape(-1,2)
            self.animation['animal'] = deque([ (3, self.trajectory_teleport(trajectory)) ])
        '''
        for pos in trajectory:
            self.jov.teleport(prefix='console', target_pos=pos)
            yield



#------------------------------------------------------------------------------
# one cue task
#------------------------------------------------------------------------------
class one_cue_task(Task):

    def __init__(self, jov):

        fsm = {
                '1cue': {'_dcue_000': ['1cue', self.goal_cue_touched, 'reward']} 
              }

        super(one_cue_task, self).__init__(fsm, jov)

        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_001')

        @self.ani.connect
        def on_animation_finish(animation_name):
            if animation_name == 'bury':
                self.reset()

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        super(one_cue_task, self).reset()
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_goal   = _cue_generate_2d_maze(self._corrd_animal) 
        self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)), (30, self.vibrate('_dcue_000')) ])
        self.state = '1cue'

    def goal_cue_touched(self, args):
        self.log.info(args)
        self.jov.reward(self.reward_time)
        self.transition_enable.behave = False
        self.animation['_dcue_000'] = deque([ (4, self.bury('_dcue_000')) ])



#------------------------------------------------------------------------------
# one cue moving task
#------------------------------------------------------------------------------
class one_cue_moving_task(Task):

    def __init__(self, jov):

        fsm = {
                '1cue': {'_dcue_000': ['1cue', self.goal_cue_touched, 'reward']} 
              }

        super(one_cue_moving_task, self).__init__(fsm, jov)

        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_001')

        @self.ani.connect
        def on_animation_finish(animation_name):
            if animation_name == 'bury':
                self.reset()

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        super(one_cue_moving_task, self).reset()
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_goal   = _cue_generate_2d_maze(self._corrd_animal) 
        # self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)), (4, self.wander('_dcue_000', direction='x')) ])
        self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)), (2, self.wander('_dcue_000')) ])
        self.state = '1cue'

    def goal_cue_touched(self, args):
        self.log.info(args)
        self.jov.reward(self.reward_time)
        self.transition_enable.behave = False
        self.animation['_dcue_000'] = deque([ (4, self.bury('_dcue_000')) ])



#------------------------------------------------------------------------------
# two cue task
#------------------------------------------------------------------------------
class two_cue_task(Task):

    def __init__(self, jov):
        # goal cue: 000, guide cue: 001
        fsm = {
                '2cue': { '_dcue_000': ['2cue', self.warn, 'touch wrong cue'],    '_dcue_001': ['1cue', self.guide_cue_touched, 'right cue'] }, 
                '1cue': { '_dcue_000': ['2cue', self.goal_cue_touched, 'reward'], '_dcue_001': ['1cue', self.guide_cue_touched, 'wrong cue'] } 
              }
        super(two_cue_task, self).__init__(fsm, jov)

        @self.ani.connect
        def on_animation_finish(animation_name):
            if animation_name == 'trajectory_teleport':
                pass
            elif animation_name == 'bury':
                self.reset()

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        super(two_cue_task, self).reset()
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_guide  = _cue_generate_2d_maze(self._corrd_animal) 
        self._coord_goal   = _cue_generate_2d_maze(self._corrd_animal, self._coord_guide)
        self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)) ])
        self.animation['_dcue_001'] = deque([ (3, self.parachute('_dcue_001', self._coord_guide)), (30, self.vibrate('_dcue_001')) ])
        self.state = '2cue'

    def warn(self, args):
        # TODO: give sound
        self.log.info(args)

    def guide_cue_touched(self, args):
        self.log.info(args)
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_001')
        self.animation['_dcue_000'] = deque([ (30, self.vibrate('_dcue_000')) ]) 

    def goal_cue_touched(self, args):
        self.log.info(args)
        # self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_000')
        self.jov.reward(self.reward_time)
        self.transition_enable.behave = False
        # trajectory = np.arange(-99,100).repeat(2).reshape(-1,2)
        # self.animation['animal'] = deque([ (1, self.trajectory_teleport(trajectory)) ])
        self.animation['_dcue_000'] = deque([ (2, self.bury('_dcue_000')) ])
        # self.reset()


#------------------------------------------------------------------------------
# empty task
#------------------------------------------------------------------------------
class empty_task(Task):
    """docstring for empty_task"""
    def __init__(self, arg):
        super(empty_task, self).__init__()
        self.arg = arg



if __name__ == '__main__':
    from playground.base import Jovian
    jov = Jovian()
    task = two_cue_task('2cue', jov)
    jov.emit('touch', args=(0, (0,0,0)))


