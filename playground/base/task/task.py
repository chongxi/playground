import numpy as np
from ..jovian import Jovian
from spiketag.utils import EventEmitter
import time
import torch
from itertools import chain
from collections import deque, namedtuple
from abc import abstractmethod


def _cue_generate_2d_maze(maze_border, *args):
    '''generate a cue position that is not going to be near to the _pos in args
       low, high are border condition for cue generation
       function should return < 100us
    '''
    radius = 50 

    while True:
        break_condition = True
        # pos = np.random.randint(low=-45, high=45, size=(2,))
        # rectangle maze
        x = np.random.randint(low=maze_border[0][0]+5, high=maze_border[0][1]-5, size=(1,))[0]
        y = np.random.randint(low=maze_border[1][0]+5, high=maze_border[1][1]-5, size=(1,))[0]
        pos = np.array([x, y])
        for _pos in args:
            if np.linalg.norm(pos - np.array(_pos)) < radius:
                break_condition = False 
        if np.linalg.norm(pos) > 95:
            break_condition = False 
        if break_condition:
            break
    return pos


class TouchEvent(EventEmitter):
    """docstring for Event"""
    def __init__(self, _id, _coord):
        super(TouchEvent, self).__init__()
        self.type = 'touch'
        self.what = _id
        self.where = _coord

class AlignEvent(EventEmitter):
    """docstring for Event"""
    def __init__(self, _id, _coord):
        super(AlignEvent, self).__init__()
        self.type = 'align'
        self.what = _id
        self.where = _coord

        
class Task(object):
    '''Every task has the same jov to report event
       Every task has different fsm is defined in child task class
       Key for the task is the fsm design, which is a dictionary:
       At the very core: when jovian report an event, with its type, what and where
       ```
        next_state, func, args = self.fsm[self.state][event.type + '@' + event.what]
        func(args)
       ```
    ''' 
    def __init__(self, fsm, jov):
        # jovian first
        self.jov = jov
        # init pos for all the cues, after this, self.jov.shared_cue_dict will have all the cues
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_000')
        self.jov.teleport(prefix='model', target_pos=(2000, 2000, 1000), target_item='_dcue_001')

        self.log = jov.log
        self.fsm = fsm 
        self.transition_enable = namedtuple("transition_enable", ['behave', 'env', 'ephys']) # verbose = False
        self.transition_enable.behave, self.transition_enable.ephys = True, False
        self.ani = EventEmitter()
        self.animation = {}  # {"_dcue_000": deque([ (4, parachute), (60, vibrate) ]), "_dcue_111": deque([ (2, animation111) ])}

        self.reward_time = 1.0
        self.BMI_enable = False

        # initial reset only after jov process start 
        @self.jov.connect
        def on_start():
            self.reset()  

        # fsm will use jov real-time event to update
        @self.jov.connect
        def on_touch(args):
            self.touch_id, self.coord = args
            self.event = TouchEvent(self.touch_id, self.coord)
            self.on_event(self.event) 

        # fsm will use jov real-time event to update
        @self.jov.connect
        def on_align(args):
            self.align_id, self.coord = args
            self.event = AlignEvent(self.align_id, self.coord)
            self.on_event(self.event) 

        # use jov frame update to do the animation
        # check the doc of "animation effect"
        @self.jov.connect
        def on_frame():
            for _cue_name, _action_queue in self.animation.items():
                try: 
                    _pace, _animation = _action_queue[0]
                    if self.jov.cnt % _pace == 0:
                        try:
                            _animation.__next__()
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
        if self.transition_enable.behave:
            self.log.info('state: {}, {}: {}@{}'.format(self.state, event.type, event.what, event.where))
            try:
                next_state, func, args = self.fsm[self.state][event.type + '@' + event.what]
                func(args)
                self.state = next_state
                self.log.info('state: {}'.format(self.state))
            except Exception as e:
                self.log.warn(f'task event processing error: {e}')


    #------------------------------------------------------------------------------
    # animation effect
    # each animation is a deque object of tuples, each tuple is (pace, animation)
    # the `pace` is the number of frame between each animation to be played
    # the `animation` is a function that will be called in the loop
    #------------------------------------------------------------------------------

    def parachute(self, cue_name, pos):
        ''' usage:
            self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)) ])
        '''
        for z in range(100,-1,-2):
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  z], target_item=cue_name)
            yield

    def bury(self, cue_name):
        self.transition_enable.behave = False
        for z in range(0, -100, -2):
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  z], target_item=cue_name)
            yield

    def vibrate(self, cue_name):
        ''' usage:
            self.animation['_dcue_001'] = deque([ (3, self.parachute('_dcue_001', self._coord_guide)), (30, self.vibrate('_dcue_001')) ])
        '''
        for z in range(5000000):
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  5], target_item=cue_name)
            yield
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            self.jov.teleport(prefix='model', target_pos=[pos[0],  pos[1],  0], target_item=cue_name)
            yield            

    def wander(self, cue_name, direction='x'):
        for i in range(20000):
            pos = self.jov._to_maze_coord(self.jov.shared_cue_dict[cue_name])
            print('pos',pos)
            pos = pos[:2].astype(float)
            radius = np.linalg.norm(pos)
            print(radius)
            if radius > 95:  # if direction=='x':
                print('outer loop')
                _x = pos[0]
                while _x>=-95:
                    _x -= 5
                    self.jov.teleport(prefix='model', target_pos=[_x,  pos[1],  0], target_item=cue_name)
                    yield
                while _x<=95:
                    _x += 5
                    self.jov.teleport(prefix='model', target_pos=[_x,  pos[1],  0], target_item=cue_name)
                    yield
            else: # direction=='circular':
                print('inner loop')
                _x, _y  = pos
                _delta = 0.0
                print(pos)
                theta = np.arctan(_y/_x)
                print(theta)
                _delta = np.pi/64
                while True:
                    # theta  += _delta
                    # self.target_x, self.target_y = radius*np.cos(theta),  radius*np.sin(theta)
                    # self.head_v = np.arccos((self.target_y-self.last_y)/(self.target_x-self.last_x))*180/np.pi
                    # self.jov.teleport(prefix='model', target_pos=[radius*np.cos(theta),  radius*np.sin(theta),  0], target_item=cue_name)
                    # self.jov.teleport(prefix='console', target_pos=[self.target_x, self.target_y,  0])
                    # self.jov.teleport(prefix='console', target_pos=[self.target_x, self.target_y,  0], head_direction=90-theta*180/np.pi)

                    # test rotation
                    hd = self.jov.cnt[0]%360 - 180
                    print(hd)
                    # self.jov.info('head direction {}\n'.format(hd))
                    self.jov.teleport(prefix='console', target_pos=[10, 20,  0], head_direction=hd)
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


    def bmi_control(self, prefix='console', cue_name=None):
        ''' usage:
            self.animation['_dcue_001'] = deque([ (3, self.parachute('_dcue_001', self._coord_guide)), (30, self.vibrate('_dcue_001')) ])
        '''
        while True:
            if self.BMI_enable:
                self.jov.teleport(prefix=prefix, 
                                  target_pos=[self.jov.bmi_pos[0], self.jov.bmi_pos[1], 5], 
                                  head_direction=self.jov.bmi_hd[0], 
                                  target_item=cue_name)
            yield


#------------------------------------------------------------------------------
# one cue task
#------------------------------------------------------------------------------
class one_cue_task(Task):

    def __init__(self, jov):

        fsm = {
                '1cue': {'touch@_dcue_000': ['1cue', self.goal_cue_touched, 'reward']} 
              }

        super(one_cue_task, self).__init__(fsm, jov)

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
        self._coord_goal   = _cue_generate_2d_maze(self.jov.maze_border, self._corrd_animal) 
        self.animation['_dcue_000'] = deque([ (4, self.parachute('_dcue_000', self._coord_goal)), (30, self.vibrate('_dcue_000')) ])
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
                '1cue': {'touch@_dcue_000': ['1cue', self.goal_cue_touched, 'reward']} 
              }

        super(one_cue_moving_task, self).__init__(fsm, jov)

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
        self._coord_goal   = _cue_generate_2d_maze(self.jov.maze_border, self._corrd_animal) 
        self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)), (3, self.wander('_dcue_000')) ])
        # self.animation['_dcue_001'] = deque([ (3, self.parachute('_dcue_001', self._coord_goal)), (4, self.wander('_dcue_001', direction='x')) ])
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
                '2cue': { 'touch@_dcue_000': ['2cue', self.warn, 'touch wrong cue'],    'touch@_dcue_001': ['1cue', self.guide_cue_touched, 'right cue'] }, 
                '1cue': { 'touch@_dcue_000': ['2cue', self.goal_cue_touched, 'reward'], 'touch@_dcue_001': ['1cue', self.guide_cue_touched, 'wrong cue'] } 
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
        self._coord_guide  = _cue_generate_2d_maze(self.jov.maze_border, self._corrd_animal) 
        self._coord_goal   = _cue_generate_2d_maze(self.jov.maze_border, self._corrd_animal, self._coord_guide)
        self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)),  (30, self.vibrate('_dcue_000')) ])
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


#------------------------------------------------------------------------------
# RING
#------------------------------------------------------------------------------
class RING(Task):

    def __init__(self, jov):

        fsm = {
                '1cue': {'align@_dcue_000': ['1cue', self.goal_cue_touched, 'reward']} 
              }

        super(RING, self).__init__(fsm, jov)

        @self.ani.connect
        def on_animation_finish(animation_name):
            if animation_name == 'bury':
                self.reset()

    def _bmi_control(self, prefix='console', cue_name=None):
        ''' usage:
            self.animation['_dcue_001'] = deque([ (3, self.parachute('_dcue_001', self._coord_guide)), (30, self.vibrate('_dcue_001')) ])
        '''
        while True:
            self.jov.teleport(prefix=prefix, 
                              target_pos=[0, 0, 5], 
                              head_direction=self.jov.bmi_hd[0] + self.jov.current_hd[0],  # current_hd is from rotation encoder
                              target_item=cue_name)
            yield

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        super(RING, self).reset()
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_goal   = _cue_generate_2d_maze(self.jov.maze_border, self._corrd_animal) 
        self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)), (3, self._bmi_control('console')) ])
        self.state = '1cue'

    def goal_cue_touched(self, args):
        self.log.info(args)
        self.jov.reward(self.reward_time)
        self.transition_enable.behave = False
        self.animation['_dcue_000'] = deque([ (4, self.bury('_dcue_000')) ])


#------------------------------------------------------------------------------
# JEDI
#------------------------------------------------------------------------------
class JEDI(Task):

    def __init__(self, jov):
        '''
        ('_dcue_000', '_dcue_001') meaning these two cue generate a touch event (they collide)
        In this task, the agent controls the _dcue_001 to touch _dcue_000
        once this happen, state transition from `1cue` to `1cue` and goal_cue_touched function is triggered
        in which, reward is given and `_dcue_000` goes to bury. 
        once bury animation finished, task will reset() and a new trial start (trasition_enable becomes True)
        '''

        fsm = {
                '1cue': { 'touch@_dcue_000->_dcue_001': ['1cue', self.goal_cue_touched, 'reward'] } 
              }

        super(JEDI, self).__init__(fsm, jov)

        self.BMI_enable = True
        self.reward_time = 0.01

        #------------------------------------------------------------------------------
        # core of JEDI: teleport cue(`_dcue_001`) when bmi_decode event happens
        @self.jov.connect
        def on_bmi_update(pos):
            if self.jov.cnt > self._last_cnt:
                self.jov.teleport(prefix='model', target_pos=(pos[0], pos[1], 15), target_item='_dcue_001')
            self._last_cnt = self.jov.cnt
        #------------------------------------------------------------------------------

        @self.ani.connect
        def on_animation_finish(animation_name):
            if animation_name == 'bury':
                self.reset()

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        super(JEDI, self).reset()
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_goal   = _cue_generate_2d_maze(self.jov.maze_border, self._corrd_animal) 
        self.animation['_dcue_000'] = deque([ (3, self.parachute('_dcue_000', self._coord_goal)),  
                                              (32,self.vibrate('_dcue_000'))  ])
        self.animation['_dcue_001'] = deque([ (2, self.bmi_control('model','_dcue_001')) ])
        self.BMI_enable = True
        self.log.info('BMI control enabled')
        self.state = '1cue'

    def goal_cue_touched(self, args):
        self.jov.reward(self.reward_time)


#------------------------------------------------------------------------------
# JUMPER
#------------------------------------------------------------------------------
class JUMPER(Task):

    def __init__(self, jov):

        fsm = {
                '1cue': {'touch@_dcue_000': ['1cue', self.goal_cue_touched, 'reward']} 
              }

        super(JUMPER, self).__init__(fsm, jov)

        self.BMI_enable = True

        #------------------------------------------------------------------------------
        # core of JUMPER: teleport itself when bmi_decode event happens
        @self.jov.connect
        def on_bmi_update(pos):
            self.jov.teleport(prefix='console', target_pos=(pos[0], pos[1], 15))
        #------------------------------------------------------------------------------

        @self.ani.connect
        def on_animation_finish(animation_name):
            if animation_name == 'bury':
                self.reset()

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        super(JUMPER, self).reset()
        self._corrd_animal = self.jov._to_maze_coord(self.current_pos)[:2]
        self._coord_goal   = _cue_generate_2d_maze(self.jov.maze_border, self._corrd_animal) 
        self.animation['_dcue_000'] = deque([ (4, self.parachute('_dcue_000', self._coord_goal)), (3, self.bmi_control('console')) ])
        self.BMI_enable = True
        self.log.info('BMI control enabled')
        self.state = '1cue'

    def goal_cue_touched(self, args):
        self.log.info(args)
        self.jov.reward(self.reward_time)
        self.transition_enable.behave = False
        self.BMI_enable = False
        self.log.info('BMI control disabled')
        self.animation['_dcue_000'] = deque([ (4, self.bury('_dcue_000')) ])


if __name__ == '__main__':
    from playground.base import Jovian
    jov = Jovian()
    task = two_cue_task('2cue', jov)
    jov.emit('touch', args=(0, (0,0,0)))

