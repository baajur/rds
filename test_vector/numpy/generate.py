import sys
import os
import numpy as np

directory = sys.argv[1]
if os.path.isdir(directory) == False:
    os.mkdir(directory)

array1d = [3*i for i in range(0,3)]
array2d = [[3*i + 5*j for j in range(0,3)] for i in range(0,3)]
array3d = [[[3*i + 5*j + 7*k for k in range(0,3)] for j in range(0,3)] for i in range(0,3)]

types = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64'];

for t in types:
    dt = np.dtype(t)
    for bo in ['<', '>']:
        dt = np.dtype(bo + dt.str[1:])
        np.save(directory + '/1d_' + bo + t, np.array(array1d, dtype=dt, order='C'))
        np.save(directory + '/2d_' + bo + t, np.array(array2d, dtype=dt, order='C'))
        np.save(directory + '/3d_' + bo + t, np.array(array3d, dtype=dt, order='C'))
        np.save(directory + '/fortran_1d_' + bo + t, np.array(array1d, dtype=dt, order='F'))
        np.save(directory + '/fortran_2d_' + bo + t, np.array(array2d, dtype=t, order='F'))
        np.save(directory + '/fortran_3d_' + bo + t, np.array(array3d, dtype=t, order='F'))
