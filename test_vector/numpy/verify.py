import sys
from os import listdir
from os.path import join
import numpy as np

directory=sys.argv[1]

for f in listdir(directory):
    if f.endswith('.npy'):
        array = np.load(join(directory,f))
        if array.ndim == 1:
            for i in range(array.shape[0]):
                if array[i] != i * 3:
                    print('{}: Fail: {} != {}'.format(f, array[i], i * 3))
                    sys.exit(1)
        elif array.ndim == 2:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    if array[i][j] != i * 3 + j * 5:
                        print('{}: Fail: {} != {}'.format(f, array[i][j], i * 3 + j * 5))
                        sys.exit(1)
        elif array.ndim == 3:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    for k in range(array.shape[2]):
                        if array[i][j][k] != i * 3 + j * 5 + k * 7:
                            print('{}: Fail: {} != {}'.format(f, array[i][j][k], i * 3 + j * 5 + k * 7))
                            sys.exit(1)
        else:
            print('{}: Fail'.format(f))
            sys.exit(1)
