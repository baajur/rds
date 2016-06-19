use array::{NDArray, NDData, NDDataMut, NDSliceable, NDSliceableMut};

#[test]
fn indexing() {
    let mut array = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                array[&[i, j, k]] = (i * 3 + j * 5 + k * 7) as f64;
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                assert!(array[&[i,j,k]] == (i * 3 + j * 5 + k * 7) as f64);
            }
        }
    }
}

#[test]
#[should_panic]
fn indexing_wrongdim() {
    let mut array = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    array[&[1,2]] = 1.0;
}

#[test]
#[should_panic]
fn indexing_outofbound() {
    let mut array = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    array[&[1,4,1]] = 1.0;
}

#[test]
fn slicing() {
    let mut array = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                array[&[i, j, k]] = (i * 3 + j * 5 + k * 7) as f64;
            }
        }
    }
    for i in 0..3 {
        let slice2d = array.slice(&[i]);
        for j in 0..3 {
            let slice1d = slice2d.slice(&[j]);
            for k in 0..3 {
                assert!(slice1d[&[k]] == (i * 3 + j * 5 + k * 7) as f64);
            }
        }
    }
}

#[test]
fn slicing_mut() {
    let mut array = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            let mut slice1d = array.slice_mut(&[i, j]);
            for k in 0..3 {
                slice1d[&[k]] = (i * 3 + j * 5 + k * 7) as f64;
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                assert!(array[&[i,j,k]] == (i * 3 + j * 5 + k * 7) as f64);
            }
        }
    }
}
#[test]
fn copy() {
    let mut array = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                array[&[i, j, k]] = (i * j * k) as f64;
            }
        }
    }
    let array2 = NDArray::copy(&array.slice(&[2]));
    for i in 0..3 {
        for j in 0..3 {
            assert!(array2[&[i,j]] == (i * j * 2) as f64);
        }
    }
}

#[test]
fn from_slice() {
    let mut array = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                array[&[i, j, k]] = (i * j * k) as f64;
            }
        }
    }
    let array2 = NDArray::from_slice(&[3, 3, 3], array.get_data());
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                assert!(array2[&[i,j,k]] == (i * j * k) as f64);
            }
        }
    }
}

#[test]
fn reshape() {
    let mut array = NDArray::<f64>::new(&[3], 0.0);
    for i in 0..3 {
        array[&[i]] = i as f64;
    }
    array.reshape(&[1,3]);
    for i in 0..3 {
        assert!(array[&[0,i]] == i as f64);
    }
}

#[test]
#[should_panic]
fn reshape_invalid() {
    let mut array = NDArray::<f64>::new(&[3], 0.0);
    for i in 0..3 {
        array[&[i]] = i as f64;
    }
    array.reshape(&[3,3]);
    for i in 0..3 {
        assert!(array[&[i,i]] == i as f64);
    }
}

#[test]
fn generic_transpose_2d() {
    let mut array = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                array[&[i, j, k]] = (i * 3 + j * 5 + k * 7) as f64;
            }
        }
    }
    array.slice_mut(&[1]).transpose();
    for i in 0..3 {
        for j in 0..3 {
            assert!(array[&[1,j,i]] == (3 + i * 5 + j * 7) as f64);
        }
    }
}

#[test]
fn generic_transpose_3d() {
    let mut array = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                array[&[i, j, k]] = (i * 3 + j * 5 + k * 7) as f64;
            }
        }
    }
    array.transpose();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                assert!(array[&[k,j,i]] == (i * 3 + j * 5 + k * 7) as f64);
            }
        }
    }
}

#[test]
#[should_panic]
fn generic_transpose_wrong_shape() {
    let mut array = NDArray::<f64>::new(&[3, 3, 4], 0.0);
    array.slice_mut(&[]).transpose();
}

#[test]
fn transpose_2d() {
    let mut array = NDArray::<f64>::new(&[3, 4], 0.0);
    for i in 0..3 {
        for j in 0..4 {
            array[&[i, j]] = (i * 3 + j * 5) as f64;
        }
    }
    array.transpose();
    for i in 0..3 {
        for j in 0..4 {
            assert!(array[&[j,i]] == (i * 3 + j * 5) as f64);
        }
    }
}

#[test]
fn transpose_3d() {
    let mut array = NDArray::<f64>::new(&[3, 4, 5], 0.0);
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..5 {
                array[&[i, j, k]] = (i * 3 + j * 5 + k * 7) as f64;
            }
        }
    }
    array.transpose();
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..5 {
                assert!(array[&[k,j,i]] == (i * 3 + j * 5 + k * 7) as f64);
            }
        }
    }
}

#[test]
fn equality() {
    assert!(NDArray::<u8>::new(&[3, 3, 3], 0) == NDArray::<u8>::new(&[3, 3, 3], 0));
    assert!(NDArray::<u8>::new(&[3, 3, 3], 0) != NDArray::<u8>::new(&[3, 3], 0));
    assert!(NDArray::<u8>::new(&[3, 4], 0) != NDArray::<u8>::new(&[3, 3], 0));
    assert!(NDArray::<u8>::new(&[3], 0) != NDArray::<u8>::new(&[3], 1));
    assert!(NDArray::<u8>::new(&[4, 3], 0).slice(&[1]) == NDArray::<u8>::new(&[2, 3], 0).slice(&[1]));
}

#[test]
fn assignement() {
    let mut array3d = NDArray::<f64>::new(&[3, 3, 3], 0.0);
    let mut array2d = NDArray::<f64>::new(&[3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            array2d[&[i, j]] = (i * 3 + j * 5) as f64;
        }
    }
    for i in 0..3 {
        array3d.slice_mut(&[i]).assign(&array2d);
    }
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                assert!(array3d[&[i,j,k]] == (j * 3 + k * 5) as f64);
            }
        }
    }
}
