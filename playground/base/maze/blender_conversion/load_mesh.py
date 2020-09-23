from vispy import scene
import numpy as np
from scipy.special import sph_harm
from vispy.visuals.transforms import STTransform, MatrixTransform
from vispy.io.mesh import read_mesh


canvas = scene.SceneCanvas(keys='interactive')
view = canvas.central_widget.add_view()

(vertices, faces, vertex_colors, _) = read_mesh('maze_2d_1m.obj')
gray_mat = np.repeat(np.array([0.5989, 0.6870, 0.4140]), 3).reshape(-1,3)
vertex_colors = abs(vertex_colors.dot(gray_mat))
transparency  = np.ones((vertex_colors.shape[0], 1))
print(vertex_colors)
# color = np.zeros((ys.shape[0], 4)) * np.array([0,1,1,1])
N = vertex_colors.shape[0]

# gray_vertex_colors = rgb2gray(vertex_colors)
# vertex_colors[:,0] = np.linspace(0,1,N)
# vertex_colors[:,2] = vertex_colors[::-1, 0] 
mesh = scene.visuals.Mesh(vertices, faces, vertex_colors, shading=None)
# mesh.ambient_light_color = vispy.color.Color('white')

view.camera = 'turntable'

maze_scale_factor = 100
transform = MatrixTransform()
# transform.rotate(angle=90, axis=(1, 0, 0))  
transform.scale(scale=[100,100,100,100])
transform.translate(pos=[-1309.17, -1258.14, -858.138])
mesh.transform = transform 

view.add(mesh)

canvas.show()

if __name__ == '__main__':
    canvas.app.run()
