import socket
from playground.view import maze_view

if __name__ == '__main__':
    # vertex positions of data to draw
    # pos = np.load('./trajctory.npy').astype(np.float32)
    sock_cmd = socket.create_connection(('10.102.20.26', '22223'), timeout=1)
    nav_view = maze_view()
    nav_view.load_maze('../playground/base/maze/obj/maze_2d.obj', maze_coord_file='../playground/base/maze/2dmaze_2cue_follow1_4.coords')
    nav_view.load_cue(cue_file='../playground/base/maze/obj/constraint_cue.obj', cue_name='_dcue_001')
    nav_view.load_cue(cue_file='../playground/base/maze/obj/goal_cue.obj', cue_name='_dcue_000')
    nav_view.connect(sock_cmd)
    nav_view.run()