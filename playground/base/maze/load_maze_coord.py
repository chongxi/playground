def load_maze_coord(maze_coord_file):
    object_coord = {}
    with open(maze_coord_file) as f:
        for line in (f): 
            try: 
                key, _, *value = line.rstrip('\r\n').split(' ')
                # print key, x, y, z
                if key == 'Origin':
                    object_coord[key] = (float(value[0]), float(value[1]), 0.0)
                elif key == 'border':
                    object_coord[key] = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
                else:
                    object_coord[key] = (float(value[0]), float(value[1]), float(value[2]))
            except:
                pass
    return object_coord
