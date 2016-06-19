use std::f32;
use std::f64;

use array::{NDDataMut, NDArray};
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

#[test]
fn dot_f32() {
    let mut array1 = NDArray::<f32>::new(&[3], 0.0);
    let mut array2 = NDArray::<f32>::new(&[3], 0.0);
    for i in 0..3 {
        array1[&[i]] = (i * 3) as f32;
        array2[&[i]] = (i * 5) as f32;
    }
    assert!(array1.dot(&array2) == 15.0 + 60.0) ;
}

#[test]
fn dot_f64() {
    let mut array1 = NDArray::<f64>::new(&[3], 0.0);
    let mut array2 = NDArray::<f64>::new(&[3], 0.0);
    for i in 0..3 {
        array1[&[i]] = (i * 3) as f64;
        array2[&[i]] = (i * 5) as f64;
    }
    assert!(array1.dot(&array2) == 15.0 + 60.0) ;
}

#[test]
#[should_panic]
fn dot_shouldpanic() {
    let array1 = NDArray::<f32>::new(&[5, 5], 0.0);
    let array2 = NDArray::<f32>::new(&[5, 4], 0.0);
    let _ = array1.dot(&array2);
}

#[test]
fn gemv_f32() {
    let vector1 = NDArray::<f32>::from_slice(&[3], &[1.0, 2.0, 3.0]);
    let mut vector2 = NDArray::<f32>::from_slice(&[4], &[1.5, 2.5, 3.5, 4.5]);
    let mut matrix = NDArray::<f32>::from_slice(&[4,3], &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    vector2.gemv(1.0, &matrix, &vector1, 2.0);
    assert!(vector2 == NDArray::<f32>::from_slice(&[4], &[6.0, 10.0, 11.0, 15.0]));
    // Auto transpose
    matrix.transpose();
    vector2.gemv(1.0, &matrix, &vector1, 2.0);
    assert!(vector2 == NDArray::<f32>::from_slice(&[4], &[15.0, 25.0, 26.0, 36.0]));
}

#[test]
fn gemv_f64() {
    let vector1 = NDArray::<f64>::from_slice(&[3], &[1.0, 2.0, 3.0]);
    let mut vector2 = NDArray::<f64>::from_slice(&[4], &[1.5, 2.5, 3.5, 4.5]);
    let mut matrix = NDArray::<f64>::from_slice(&[4,3], &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    vector2.gemv(1.0, &matrix, &vector1, 2.0);
    assert!(vector2 == NDArray::<f64>::from_slice(&[4], &[6.0, 10.0, 11.0, 15.0]));
    // Auto transpose
    matrix.transpose();
    vector2.gemv(1.0, &matrix, &vector1, 2.0);
    assert!(vector2 == NDArray::<f64>::from_slice(&[4], &[15.0, 25.0, 26.0, 36.0]));
}

#[test]
#[should_panic]
fn gemv_shouldpanic() {
    let vector1 = NDArray::<f64>::new(&[3], 0.0);
    let mut vector2 = NDArray::<f64>::new(&[4], 0.0);
    let matrix = NDArray::<f64>::new(&[3,3], 0.0);
    vector2.gemv(1.0, &matrix, &vector1, 2.0);
}

#[test]
fn gemm_f32() {
    let mut matrixa = NDArray::<f32>::from_slice(&[4,3], &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let mut matrixb = NDArray::<f32>::from_slice(&[3,2], &[1.0, 2.0, 2.0, 1.0, 1.0, -1.0]);
    let mut matrixc = NDArray::<f32>::from_slice(&[4,2], &[0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5]);
    matrixc.gemm(1.0, &matrixa, &matrixb, 0.5);
    assert!(matrixc == NDArray::<f32>::from_slice(&[4,2], &[3.25, 2.75, 2.75, 0.25, 2.25, 0.75, 3.75, 2.25]));
    // Auto transpose
    matrixa.transpose();
    matrixc.gemm(1.0, &matrixa, &matrixb, 1.0);
    assert!(matrixc == NDArray::<f32>::from_slice(&[4,2], &[6.25, 5.75, 5.75, 0.25, 4.25, 1.75, 7.75, 4.25]));
    // Auto transpose
    matrixa.transpose();
    matrixb.transpose();
    matrixc.gemm(1.0, &matrixa, &matrixb, 1.0);
    assert!(matrixc == NDArray::<f32>::from_slice(&[4,2], &[9.25, 8.75, 8.75, 0.25, 6.25, 2.75, 11.75, 6.25]));
    // Auto transpose
    matrixb.transpose();
    matrixc.gemm(1.0, &matrixa, &matrixb, 1.0);
    assert!(matrixc == NDArray::<f32>::from_slice(&[4,2], &[12.25, 11.75, 11.75, 0.25, 8.25, 3.75, 15.75, 8.25]));
}

#[test]
fn gemm_f64() {
    let mut matrixa = NDArray::<f64>::from_slice(&[4,3], &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let mut matrixb = NDArray::<f64>::from_slice(&[3,2], &[1.0, 2.0, 2.0, 1.0, 1.0, -1.0]);
    let mut matrixc = NDArray::<f64>::from_slice(&[4,2], &[0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5]);
    matrixc.gemm(1.0, &matrixa, &matrixb, 0.5);
    assert!(matrixc == NDArray::<f64>::from_slice(&[4,2], &[3.25, 2.75, 2.75, 0.25, 2.25, 0.75, 3.75, 2.25]));
    // Auto transpose
    matrixa.transpose();
    matrixc.gemm(1.0, &matrixa, &matrixb, 1.0);
    assert!(matrixc == NDArray::<f64>::from_slice(&[4,2], &[6.25, 5.75, 5.75, 0.25, 4.25, 1.75, 7.75, 4.25]));
    // Auto transpose
    matrixa.transpose();
    matrixb.transpose();
    matrixc.gemm(1.0, &matrixa, &matrixb, 1.0);
    assert!(matrixc == NDArray::<f64>::from_slice(&[4,2], &[9.25, 8.75, 8.75, 0.25, 6.25, 2.75, 11.75, 6.25]));
    // Auto transpose
    matrixb.transpose();
    matrixc.gemm(1.0, &matrixa, &matrixb, 1.0);
    assert!(matrixc == NDArray::<f64>::from_slice(&[4,2], &[12.25, 11.75, 11.75, 0.25, 8.25, 3.75, 15.75, 8.25]));
}

#[test]
#[should_panic]
fn gemm_shouldpanic() {
    let mut matrixa = NDArray::<f64>::from_slice(&[4,3], &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let matrixb = NDArray::<f64>::from_slice(&[3,2], &[1.0, 2.0, 2.0, 1.0, 1.0, -1.0]);
    let matrixc = NDArray::<f64>::from_slice(&[4,2], &[0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5]);
    matrixa.gemm(1.0, &matrixb, &matrixc, 0.5);
}

#[test]
fn ops_overloading() {
    let mut array1 = NDArray::<f64>::from_slice(&[3], &[2.0, 4.0, 6.0]);
    let array2 = NDArray::<f64>::from_slice(&[3], &[3.0, 6.0, 9.0]);
    let array3 = NDArray::<f64>::from_slice(&[3], &[5.0, 10.0, 15.0]);
    let matrix1 = NDArray::<f64>::from_slice(&[3,3], &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]);
    let mut matrix2 = NDArray::<f64>::copy(&matrix1);
    matrix2.transpose();
    matrix2 *= matrix1;
    array1 += array3;
    assert!(array1 == NDArray::<f64>::from_slice(&[3], &[7.0, 14.0, 21.0]));
    array1 -= array2;
    assert!(array1 == NDArray::<f64>::from_slice(&[3], &[4.0, 8.0, 12.0]));
    array1 *= matrix2;
    assert!(array1 == NDArray::<f64>::from_slice(&[3], &[28.0, 32.0, 36.0]));
}
