from vispy import scene
import numpy as np
from playground.utils import rgb2gray
from vispy.visuals.transforms import MatrixTransform
from vispy.io.mesh import read_mesh


filename = './constraint_cue.obj'

canvas = scene.SceneCanvas(keys='interactive')
view = canvas.central_widget.add_view()

(vertices, faces, vertex_colors, _) = read_mesh(filename)
vertex_colors = rgb2gray(vertex_colors) 
print(vertex_colors)
# color = np.zeros((ys.shape[0], 4)) * np.array([0,1,1,1])
N = vertex_colors.shape[0]
mesh = scene.visuals.Mesh(vertices, faces, vertex_colors, shading=None)
view.camera = 'turntable'

maze_scale_factor = 100
transform = MatrixTransform()
transform.scale(scale=[100,100,100,100])
# transform.translate(pos=[8100.4286, 8000])
mesh.transform = transform 

view.add(mesh)

canvas.show()

if __name__ == '__main__':
    canvas.app.run()
