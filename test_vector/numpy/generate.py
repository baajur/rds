import numpy as np

array1d = [3*i for i in range(0,3)]
array2d = [[3*i + 5*j for i in range(0,3)] for j in range(0,3)]
array3d = [[[3*i + 5*j + 7*k for i in range(0,3)] for j in range(0,3)] for k in range(0,3)]

types = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64'];

for t in types:
    np.save('1d_' + t, np.array(array1d, dtype=t, order='C'))
    np.save('2d_' + t, np.array(array2d, dtype=t, order='C'))
    np.save('3d_' + t, np.array(array3d, dtype=t, order='C'))
    np.save('fortran_1d_' + t, np.array(array1d, dtype=t, order='F'))
    np.save('fortran_2d_' + t, np.array(array2d, dtype=t, order='F'))
    np.save('fortran_3d_' + t, np.array(array3d, dtype=t, order='F'))
