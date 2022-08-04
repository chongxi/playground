## filename is the obj file output from blender maze (file->export->obj: y forward, -z up, selection only)


filename = "complex_maze3 2.obj"
ofilename = 'maze_2d.obj' 

f = open(filename)
ofile = open(ofilename, "w")

line = "#SHOULD NOT BE HERE THIS LINE IS"
while line:
    line = f.readline()
    if len(line) == 0: continue
    if line[0] == "f":
        # if the number of entries following the f is more than 3,
        # write multiple face lines as all triangles
        faces = []
        oldindex = 1
        index = line.find(" ", 2)
        while index != -1:
            faces.append(line[oldindex+1:index])
            oldindex = index
            index = line.find(" ", index+1)
        # I'm assuming no faces at end of line
        faces.append(line[oldindex+1:len(line)-1])
        #print faces
        start = 1
        while start + 1 < len(faces):
            ofile.write("f " + faces[0] + " " + faces[start] + " " + faces[start+1] + "\n")
            start += 1
    else:
        ofile.write(line)
ofile.close()
f.close()
