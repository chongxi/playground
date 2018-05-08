import numpy as np
from vispy import app, scene, visuals, gloo
from vispy.color import Color
from vispy.visuals.transforms import STTransform

white = Color("#ecf0f1")
gray = Color("#121212")
red = Color("#e74c3c")
blue = Color("#2980b9")
orange = Color("#e88834")


def star(inner=0.5, outer=1.0, n=5):
    R = np.array([inner, outer] * n)
    T = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    P = np.zeros((2 * n, 3))
    P[:, 0] = R * np.cos(T)
    P[:, 1] = R * np.sin(T)
    return P


def rec(left=-15, right=15, bottom=-25, top=25):
    P = np.zeros((4, 3))
    R = np.array([[left,  bottom],
                  [right, bottom],
                  [right, top   ],
                  [left,  top   ]])
    P[:, :2] = R
    return P



class shank(object):
    def __init__(self, pos):
        self.pos = pos

class probe_geometry(object):
    """docstring for probe_geometry"""
    def __init__(self, shanks):
        super(probe_geometry, self).__init__()
        self.shanks = shanks


class probe_view(scene.SceneCanvas):
    '''probe view
    '''
    def __init__(self):
        scene.SceneCanvas.__init__(self, keys=None, title='probe view')
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.electrode_pads = scene.visuals.Markers(parent=self.view.scene)
        self.electrode_text = scene.visuals.Text(parent=self.view.scene)
        # self.electrode_edge = scene.visuals.Line(antialias=False, method='gl', color=(1, 1, 1, 0.2), parent=self.view.scene)
        self.electrode_poly = [] 
        

    def set_data(self, pos):
        # self.prb = probe
        self.pos = pos
        self.electrode_pads.set_data(self.pos, symbol='square', size=17)
        self.electrode_text.text = [str(i) for i in range(len(pos))] 
        self.electrode_text.pos  = pos
        self.electrode_text.font_size = 6
        edges = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],
                          [4,5],[4,6],[4,7],[5,6],[5,7],[6,7]])
        color = np.ones((pos.shape[0], 4))
        color[:4,:] = np.array([1,0,0,1])
        self.electrode_edge = scene.visuals.Line(pos=pos, connect=edges, antialias=False, method='gl',
                                                 color=color, parent=self.view.scene)

        # P = rec(-15,15,-10,30)
        # print P
        # n = 1
        # for i in range(n):
        #     # c = i / float(n)
        #     # x, y = np.random.uniform(-1, +1, 2)
        #     x = 0 
        #     y = i*40
        #     s = 1 
        #     self.electrode_poly.append(scene.visuals.Polygon(P*s + (x, y, 0), color=(1, 0, 0, 0.5), 
        #                                                                       border_color=(1,1,1,1), parent=self.view.scene))

        self.view.camera.set_range([-100,100])

    def imap(self, mouse_pos):
        tr = self.view.scene.transform
        Point = tr.imap(mouse_pos)[:2]
        return Point

    def on_key_press(self, e):
        if e.text == 'r':
            self.view.camera.set_range([-100,100])

    def on_mouse_release(self, e):
        if e.button == 1:
            print self.imap(e.pos)


    def run(self):
        self.show()
        self.app.run()


if __name__ == '__main__':
    # prb = 'bow-tie'
    prb_view = probe_view()
    nCh = 64
    y_pos = np.linspace(0,600,32).reshape(-1,1)
    x_pos = np.ones_like(y_pos) * -10
    l_shank = np.hstack((x_pos, y_pos))  
    r_shank = np.hstack((-x_pos, y_pos))
    pos = np.empty((l_shank.shape[0] + r_shank.shape[0], 2))
    pos[::2] = l_shank
    pos[1::2] = r_shank
    prb_view.set_data(pos)
    prb_view.run()
