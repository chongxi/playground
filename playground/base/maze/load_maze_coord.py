def load_maze_coord(maze_coord_file):
    object_coord = {}
    with open(maze_coord_file) as f:
        for line in (f): 
            try: 
                key, _, x, y, z = line.rstrip('\r\n').split(' ')
                # print key, x, y, z
                if key == 'Origin':
                    object_coord[key] = (float(x),float(y),0.0)
                else:
                    object_coord[key] = (float(x),float(y),float(z))
            except:
                pass
    return object_coord
