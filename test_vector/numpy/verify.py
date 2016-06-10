from os import listdir
from os.path import join
from sys import exit
import numpy as np

path='test_vector/numpy/out'

for f in listdir(path):
    if f.endswith('.npy'):
        array = np.load(join(path,f))
        if array.ndim == 1:
            for i in range(array.shape[0]):
                if array[i] != i * 3:
                    print('{}: Fail: {} != {}'.format(f, array[i], i * 3))
        elif array.ndim == 2:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if array[i][j] != i * 3 + j * 5:
                        print('{}: Fail: {} != {}'.format(f, array[i][j], i * 3 + j * 5))
        elif array.ndim == 3:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    for k in range(array.shape[2]):
                        if array[i][j][k] != i * 3 + j * 5 + k * 7:
                            print('{}: Fail: {} != {}'.format(f, array[i][j][k], i * 3 + j * 5 + k * 7))
        else:
            print('{}: Fail'.format(f))
            exit(1)
        print('{}: Ok'.format(f))
