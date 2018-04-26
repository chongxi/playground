import numpy as np
from vispy import app, gloo, scene, visuals
from vispy.geometry import create_sphere
from vispy.visuals import CompoundVisual, MeshVisual
from vispy.io.mesh import read_mesh
from ..utils import rgb2gray, EventEmitter
from ..base.maze import load_maze_coord
import time


scale = 100

def _to_maze_coord(pos, origin, scale=scale):
    '''tranform back to origin (0,0,0)
    '''
    return (pos-origin)/scale

def _to_jovian_coord(pos, origin, scale=scale):
    return (pos*scale)+origin


class LineVisual(visuals.Visual):
    """Example of a very simple GL-line visual.
    
    This shows the minimal set of methods that need to be reimplemented to 
    make a new visual class.
    
    """
    def __init__(self, pos=None, color=(1, 0, 1, 1)):
        vcode = """
        attribute vec2 pos;
        attribute vec4 color;
        varying vec4 v_color;
        
        void main() {
            gl_Position = $transform(vec4(pos, 0, 1)); 
            gl_PointSize = 10;
            v_color = color;
        }
        """
        
        fcode = """
        varying vec4 v_color;

        void main() {
            gl_FragColor = v_color;
        }
        """
        
        visuals.Visual.__init__(self, vcode=vcode, fcode=fcode)
        
        self.pos_buf = gloo.VertexBuffer()
        self.color_buf = gloo.VertexBuffer()
        
        # The Visual superclass contains a MultiProgram, which is an object
        # that behaves like a normal shader program (you can assign shader
        # code, upload values, set template variables, etc.) but internally
        # manages multiple ModularProgram instances, one per view.
        
        # The MultiProgram is accessed via the `shared_program` property, so
        # the following modifications to the program will be applied to all 
        # views:
        self.shared_program['pos'] = self.pos_buf
        self.shared_program['color'] = self.color_buf
        self._need_upload = False
        
        # Visual keeps track of draw mode, index buffer, and GL state. These
        # are shared between all views.
        self._draw_mode = 'line_strip'
        self.set_gl_state('translucent', depth_test=False)
        
        if pos is not None:
            self.set_data(pos)
            
    def set_data(self, pos, color=None):
        self.unfreeze()
        self._pos = pos
        if color is None:
            self._color = np.ones((self._pos.shape[0], 4))
        else:
            self._color = color
        self._need_upload = True

    def _prepare_transforms(self, view=None):
        view.view_program.vert['transform'] = view.transforms.get_transform()

    def _prepare_draw(self, view=None):
        """This method is called immediately before each draw.
        
        The *view* argument indicates which view is about to be drawn.
        """
        if self._need_upload:
            # Note that pos_buf is shared between all views, so we have no need
            # to use the *view* argument in this example. This will be true
            # for most visuals.
            # self.pos_buf.set_data(self._pos)
            self.shared_program['pos']   = self._pos
            self.shared_program['color'] = self._color
            # self.color_buf.set_data(self._color)
            self._need_upload = False



class MazeVisual(CompoundVisual):
    """Visual that displays a maze
    """
    def __init__(self, maze_file,  maze_coord_file=None, edge_color=None, **kwargs):
        (vertices, faces, vertex_colors, _) = read_mesh(maze_file)
        vertex_colors = rgb2gray(vertex_colors)
        self._mesh = scene.visuals.Mesh(vertices, faces, vertex_colors, shading=None) #color='gray'
        self._border = scene.visuals.Mesh()
        self._coord = load_maze_coord(maze_coord_file)

        CompoundVisual.__init__(self, [self._mesh, self._border], **kwargs)
        self.mesh.set_gl_state(polygon_offset_fill=True,
                               polygon_offset=(1, 1), depth_test=True)


    @property
    def mesh(self):
        return self._mesh

    @property
    def border(self):
        return self._border

    @property
    def coord(self):
        '''Return the coord (read from coord file generated by Jovian)
        '''
        return self._coord 




class AnimalVisual(CompoundVisual, EventEmitter):
    """Visual that displays animal as a sphere

    """
    def __init__(self, radius=1.0, cols=30, rows=30, depth=30, subdivisions=3,
                 method='latitude', vertex_colors=None, face_colors=None,
                 color=(0.5, 0.5, 1, 1), edge_color=None, **kwargs):

        EventEmitter.__init__(self)

        self._origin = np.array([0,0,0])
        self._pos    = np.array([0,0,0])
        self._scale_factor = scale

        mesh = create_sphere(cols, rows, depth, radius=radius,
                             subdivisions=subdivisions, method=method)

        self._mesh = MeshVisual(vertices=mesh.get_vertices(),
                                faces=mesh.get_faces(),
                                vertex_colors=vertex_colors,
                                face_colors=face_colors, color=color)
        if edge_color:
            self._border = MeshVisual(vertices=mesh.get_vertices(),
                                      faces=mesh.get_edges(),
                                      color=edge_color, mode='lines')
        else:
            self._border = MeshVisual()

        CompoundVisual.__init__(self, [self._mesh, self._border], **kwargs)
        self.mesh.set_gl_state(polygon_offset_fill=True,
                               polygon_offset=(1, 1), depth_test=True)

    @property
    def mesh(self):
        """The vispy.visuals.MeshVisual that used to fil in.
        """
        return self._mesh

    @property
    def border(self):
        """The vispy.visuals.MeshVisual that used to draw the border.
        """
        return self._border

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, v):
        self._origin = np.array(v)

    @property
    def pos(self):
        return _to_maze_coord(self._pos, self._origin, self._scale_factor)

    @pos.setter
    def pos(self, v):
        self.emit('move', target_pos=v)
        # self._pos = _to_jovian_coord(np.array(v), self._origin, self._scale_factor)
        # self._transform.translate = self._pos

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, tr):
        self._transform = tr

    def scale(self, scale_factor):
        self._transform.scale = 4*[scale_factor]
        self._scale_factor = scale_factor





class CueVisual(CompoundVisual, EventEmitter):
    """Visual that displays a cue
    """
    def __init__(self, cue_file, edge_color=None, **kwargs):
        # super(CueVisual, self).__init__()
        EventEmitter.__init__(self)
        (vertices, faces, vertex_colors, _) = read_mesh(cue_file)
        vertex_colors = rgb2gray(vertex_colors)
        self._name = None
        self._mesh = scene.visuals.Mesh(vertices, faces, vertex_colors, shading=None) #color='gray'
        self._border = scene.visuals.Mesh()
        self._origin = np.array([0,0,0])
        self._center = np.array([0,0,0])
        self._pos    = np.array([0,0,0])
        self._z      = 0
        self._z_floor = 0
        self._transform = None 
        self._scale_factor = 1
        self._vib_flag = False

        CompoundVisual.__init__(self, [self._mesh, self._border], **kwargs)
        self.mesh.set_gl_state(polygon_offset_fill=True,
                               polygon_offset=(1, 1), depth_test=True)

    @property
    def name(self):
        return self._name

    @name.setter
    def origin(self, _name):
        self._name = _name

    @property
    def mesh(self):
        return self._mesh

    @property
    def border(self):
        return self._border

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, v):
        self._origin = np.array(v)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, v):
        self._center = np.array(v)
        self._z_floor = self._center[-1]

    @property
    def pos(self):
        return self._center + _to_maze_coord(self._pos, self._origin, self._scale_factor)

    @pos.setter
    def pos(self, v):
        self.emit('move', target_item=self._name, target_pos=v)
        self._pos = _to_jovian_coord((np.array(v) - self._center), self._origin, self._scale_factor)
        self._pos[-1] = -self._pos[-1]
        self._transform.translate = self._pos

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, v):
        next_pos = self.pos
        next_pos[-1] = v
        self._z = v
        self.pos = next_pos    


    @property
    def transform(self):
        return self._transform


    @transform.setter
    def transform(self, tr):
        self._transform = tr


    def scale(self, scale_factor):
        self._transform.scale = 4*[scale_factor]
        self._scale_factor = scale_factor


    def move(self, coord):
        self.pos = coord


    def floor(self):
        next_pos = self.pos
        next_pos[-1] = self._z_floor
        self.pos = next_pos        

    def elevate(self, z):
        next_pos = self.pos
        next_pos[-1] += z
        self.pos = next_pos


    def vibrate(self, v):
        if self._vib_flag:
            self.z = self._z_floor + v
        else:
            self.z = self._z_floor
        self._vib_flag = not self._vib_flag


    def toggle(self, show=True):
        if show:
            self.z = self._z_floor - 1000
        else:
            self.floor()

    def parachute(self):
        self.z = 100
        while self.z > self._z_floor:
            self.z -= 1
            time.sleep(0.1)
        



