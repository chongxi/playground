from vispy import scene
from .visuals import *

Maze = scene.visuals.create_visual_node(MazeVisual)
Line = scene.visuals.create_visual_node(LineVisual)
Animal = scene.visuals.create_visual_node(AnimalVisual)
Cue = scene.visuals.create_visual_node(CueVisual)

from .maze_view import maze_view