use std::f32;
use std::f64;

use array::NDArray;
use blas::Blas;

#[test]
fn asum_f32() {
    let array1 = NDArray::<f32>::new(&[5], 1.0);
    assert!(array1.asum() == 5.0);
    let array2 = NDArray::<f32>::new(&[5, 5], 1.0);
    assert!(array2.asum() == 25.0);
}

#[test]
fn asum_f64() {
    let array1 = NDArray::<f64>::new(&[5], 1.0);
    assert!(array1.asum() == 5.0);
    let array2 = NDArray::<f64>::new(&[5, 5], 1.0);
    assert!(array2.asum() == 25.0);
}

#[test]
fn nrm2_f32() {
    let array1 = NDArray::<f32>::new(&[5], 1.0);
    assert!(array1.nrm2() == 5.0f32.sqrt());
    let array2 = NDArray::<f32>::new(&[5, 5], 1.0);
    assert!(array2.nrm2() == 25.0f32.sqrt());
}

#[test]
fn nrm2_f64() {
    let array1 = NDArray::<f64>::new(&[5], 1.0);
    assert!(array1.nrm2() == 5.0f64.sqrt());
    let array2 = NDArray::<f64>::new(&[5, 5], 1.0);
    assert!(array2.nrm2() == 25.0f64.sqrt());
}

#[test]
fn scal_f32() {
    let mut array1 = NDArray::<f32>::new(&[5], 0.0);
    for i in 0..5 {
        array1[&[i]] = (i * 3) as f32;
    }
    array1.scal(2.0);
    for i in 0..5 {
        assert!(array1[&[i]] == (i * 6) as f32);
    }

    let mut array2 = NDArray::<f32>::new(&[5, 5], 0.0);
    for i in 0..5 {
        for j in 0..5 {
            array2[&[i,j]] = (i * 3 + j * 5) as f32;
        }
    }
    array2.scal(2.0);
    for i in 0..5 {
        for j in 0..5 {
            assert!(array2[&[i,j]] == (i * 6 + j * 10) as f32);
        }
    }
}
#[test]
fn axpy_f32() {
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

    let mut array3 = NDArray::<f32>::new(&[5, 5], 0.0);
    let mut array4 = NDArray::<f32>::new(&[5, 5], 0.0);
    for i in 0..5 {
        for j in 0..5 {
            array3[&[i,j]] = (i * 3 + j * 5) as f32;
            array4[&[i,j]] = (i * 7 + j * 9) as f32;
        }
    }
    array3.axpy(2.0, &array4);
    for i in 0..5 {
        for j in 0..5 {
            assert!(array3[&[i,j]] == (i * 17 + j * 23) as f32);
        }
    }
}

#[test]
fn axpy_f64() {
    let mut array1 = NDArray::<f64>::new(&[5], 0.0);
    let mut array2 = NDArray::<f64>::new(&[5], 0.0);
    for i in 0..5 {
        array1[&[i]] = (i * 3) as f64;
        array2[&[i]] = (i * 5) as f64;
    }
    array1.axpy(2.0, &array2);
    for i in 0..5 {
        assert!(array1[&[i]] == (i * 13) as f64);
    }

    let mut array3 = NDArray::<f64>::new(&[5, 5], 0.0);
    let mut array4 = NDArray::<f64>::new(&[5, 5], 0.0);
    for i in 0..5 {
        for j in 0..5 {
            array3[&[i,j]] = (i * 3 + j * 5) as f64;
            array4[&[i,j]] = (i * 7 + j * 9) as f64;
        }
    }
    array3.axpy(2.0, &array4);
    for i in 0..5 {
        for j in 0..5 {
            assert!(array3[&[i,j]] == (i * 17 + j * 23) as f64);
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
