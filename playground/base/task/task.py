import numpy as np
from ..jovian import Jovian


def _cue_generate_2d_maze(*args):
    '''generate a cue position that is not going to be near to the _pos in args
       low, high are border condition for cue generation
       function should return < 100us
    '''
    while True:
        break_condition = True
        pos = np.random.randint(low=-100, high=100, size=(2,))
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
        try:
            next_state, func, args = self.fsm[self.state][event]
            func(args)
            self.state = next_state
        except:
            print('Your Finite State Machine is Incomplete or Wrong')




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

        fsm = {
                '2cue': {0: ['2cue', self.warn, 'touch wrong cue'], 1: ['1cue', self.guide_cue_touched, 'right cue']}, 
                '1cue': {0: ['2cue', self.goal_cue_touched, 'reward']} 
              }

        super(two_cue_task, self).__init__(fsm, jov)

    #---------------------------------------------------------------------------------------------------
    # Every task cycle finished, you need to reset (regenerate cue based on current coordination etc..)
    #---------------------------------------------------------------------------------------------------
    def reset(self):
        _corrd_animal = self.current_pos
        _coord_guide  = _cue_generate_2d_maze(_corrd_animal) 
        _coord_goal   = _cue_generate_2d_maze(_corrd_animal, _coord_guide)
        self.jov.emit('cue', cue_id=0, func='move', args=_coord_goal)
        self.jov.emit('cue', cue_id=1, func='move', args=_coord_guide)
        self.state = '2cue'


    def warn(self, args):
        # TODO: give sound
        print(args)


    def guide_cue_touched(self, args):
        print(args)
        self.jov.emit('cue', cue_id=1, func='set_z', args=-1000)


    def goal_cue_touched(self, args):
        print(args)
        self.jov.emit('cue', cue_id=0, func='set_z', args=-1000)
        #TODO: give reward




if __name__ == '__main__':
    from playground.base import Jovian
    jov = Jovian()
    task = two_cue_task('2cue', jov)
    jov.emit('touch', args=(0, (0,0,0)))


