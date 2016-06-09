import numpy as np

array1d = [3*i for i in range(0,3)]
array2d = [[3*i + 5*j for j in range(0,3)] for i in range(0,3)]
array3d = [[[3*i + 5*j + 7*k for k in range(0,3)] for j in range(0,3)] for i in range(0,3)]

types = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64'];

for t in types:
    dt = np.dtype(t)
    for bo in ['<', '>']:
        dt = np.dtype(bo + dt.str[1:])
        np.save('test_vector/numpy/1d_' + bo + t, np.array(array1d, dtype=dt, order='C'))
        np.save('test_vector/numpy/2d_' + bo + t, np.array(array2d, dtype=dt, order='C'))
        np.save('test_vector/numpy/3d_' + bo + t, np.array(array3d, dtype=dt, order='C'))
        np.save('test_vector/numpy/fortran_1d_' + bo + t, np.array(array1d, dtype=dt, order='F'))
        np.save('test_vector/numpy/fortran_2d_' + bo + t, np.array(array2d, dtype=t, order='F'))
        np.save('test_vector/numpy/fortran_3d_' + bo + t, np.array(array3d, dtype=t, order='F'))
