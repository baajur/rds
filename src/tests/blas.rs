use array::NDArray;
use blas::Blas;

#[test]
fn axpy_1d() {
    let mut array1 = NDArray::<f32>::new(&[5], 0.0);
    let mut array2 = NDArray::<f32>::new(&[5], 0.0);
    for i in 0..5 {
        array1[&[i]] = (i * 3) as f32;
        array2[&[i]] = (i * 5) as f32;
    }
    array1.axpy(2.0, &array2);
    for i in 0..5 {
        assert!(array1[&[i]] == (i * 13) as f32);
    }
}

#[test]
fn axpy_2d() {
    let mut array1 = NDArray::<f32>::new(&[5, 5], 0.0);
    let mut array2 = NDArray::<f32>::new(&[5, 5], 0.0);
    for i in 0..5 {
        for j in 0..5 {
            array1[&[i,j]] = (i * 3 + j * 5) as f32;
            array2[&[i,j]] = (i * 7 + j * 9) as f32;
        }
    }
    array1.axpy(2.0, &array2);
    for i in 0..5 {
        for j in 0..5 {
            assert!(array1[&[i,j]] == (i * 17 + j * 23) as f32);
        }
    }
}

#[test]
#[should_panic]
fn axpy_shouldpanic() {
    let mut array1 = NDArray::<f32>::new(&[5, 5], 0.0);
    let array2 = NDArray::<f32>::new(&[5, 4], 0.0);
    array1.axpy(2.0, &array2);
}
