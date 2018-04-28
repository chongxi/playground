import numpy as np
from ..jovian import Jovian


def _cue_generate_2d_maze(*args):
    '''generate a cue position that is not going to be near to the _pos in args
       low, high are border condition for cue generation
       function should return < 100us
    '''
    while True:
        break_condition = True
        pos = np.random.randint(low=-85, high=85, size=(2,))
        for _pos in args:
            if np.linalg.norm(pos - np.array(_pos)) < 30:
               break_condition = False 
        if break_condition:
            break
    return pos



class Task(object):
    '''Every task has the same jov to report event
       Every task has different fsm is defined in child task class
    ''' 
    def __init__(self, fsm, jov):
        # jovian first
        self.jov = jov
        @self.jov.connect
        def on_touch(args):
            self.touch_id, self.coord = args
            self.on_event(self.touch_id) 

        # fsm will use jov real-time event to update
        self.fsm = fsm 
        # reset will use jov.current_pos to generate cue
        self.reset()



    @property
    def current_pos(self):
        '''shared memory of current pos of animal
        '''
        return self.jov.current_pos.numpy()


    def __call__(self, fsm):
        self.fsm = fsm
        self.reset()

    def on_event(self, event):
        # try:
        print self.state
        next_state, func, args = self.fsm[self.state][event]
        func(args)
        self.state = next_state
        print self.state
        # except:
        #     print('Your Finite State Machine is Incomplete or Wrong')




class one_cue_task(Task):

    def __init__(self, jov):

        fsm = {
                '1cue': {0: ['1cue', self.goal_cue_touched, 'reward']} 
              }

        super(one_cue_task, self).__init__(fsm, jov)

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        _corrd_animal = self.current_pos
        _coord_goal   = _cue_generate_2d_maze(_corrd_animal) 
        self.jov.emit('cue', cue_id=0, func='move', args=_coord_goal)
        self.state = '1cue'



    def guide_cue_touched(self, args):
        print(args)



    def goal_cue_touched(self, args):
        print(args)



class two_cue_task(Task):

    def __init__(self, jov):

        # goal cue: 000, guide cue: 001

        fsm = {
                '2cue': { '_dcue_000': ['2cue', self.warn, 'touch wrong cue'],    '_dcue_001': ['1cue', self.guide_cue_touched, 'right cue'] }, 
                '1cue': { '_dcue_000': ['2cue', self.goal_cue_touched, 'reward'], '_dcue_001': ['1cue', self.guide_cue_touched, 'wrong cue'] } 
              }

        super(two_cue_task, self).__init__(fsm, jov)

        self.cues_name = ['_dcue_000', '_dcue_001']  # goal cue: 000, guide cue: 001

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        _corrd_animal = self.current_pos
        _coord_guide  = _cue_generate_2d_maze(_corrd_animal) 
        _coord_goal   = _cue_generate_2d_maze(_corrd_animal, _coord_guide)
        self.jov.teleport(prefix='model', target_pos=_coord_goal, target_item='_dcue_000')
        self.jov.teleport(prefix='model', target_pos=_coord_guide, target_item='_dcue_001')
        self.state = '2cue'

    def warn(self, args):
        # TODO: give sound
        print(args)

    def guide_cue_touched(self, args):
        print(args)
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_001')


    def goal_cue_touched(self, args):
        print(args)
        self.jov.teleport(prefix='model', target_pos=(1000, 1000, 1000), target_item='_dcue_000')
        #TODO: give reward
        # self.jov.reward(100)
        self.reset()




if __name__ == '__main__':
    from playground.base import Jovian
    jov = Jovian()
    task = two_cue_task('2cue', jov)
    jov.emit('touch', args=(0, (0,0,0)))


