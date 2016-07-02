extern crate libc;

use std::fmt::Display;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign};

use types::complex::{c32, c64};
use array::{NDData, NDDataMut, NDArray, NDSlice, NDSliceMut};

const CBLAS_ROW_MAJOR : libc::c_int = 101;
#[allow(dead_code)]
const CBLAS_COL_MAJOR : libc::c_int = 102;
const CBLAS_NO_TRANS : libc::c_int = 111;
#[allow(dead_code)]
const CBLAS_TRANS : libc::c_int = 112;

#[link(name = "blas")]
#[allow(improper_ctypes)] 
extern {
    fn cblas_sasum (n : isize, x : *const libc::c_float, incx : isize) -> libc::c_float;
    fn cblas_dasum (n : isize, x : *const libc::c_double, incx : isize) -> libc::c_double;
    fn cblas_scasum (n : isize, x : *const c32, incx : isize) -> libc::c_float;
    fn cblas_dzasum (n : isize, x : *const c64, incx : isize) -> libc::c_double;

    fn cblas_snrm2 (n : isize, x : *const libc::c_float, incx : isize) -> libc::c_float;
    fn cblas_dnrm2 (n : isize, x : *const libc::c_double, incx : isize) -> libc::c_double;
    fn cblas_scnrm2 (n : isize, x : *const c32, incx : isize) -> libc::c_float;
    fn cblas_dznrm2 (n : isize, x : *const c64, incx : isize) -> libc::c_double;

    fn cblas_sscal (n : isize, a : libc::c_float, x : *mut libc::c_float, incx : isize) -> libc::c_void;
    fn cblas_dscal (n : isize, a : libc::c_double, x : *mut libc::c_double, incx : isize) -> libc::c_void;
    fn cblas_cscal (n : isize, a : *const c32, x : *mut c32, incx : isize) -> libc::c_void;
    fn cblas_zscal (n : isize, a : *const c64, x : *mut c64, incx : isize) -> libc::c_void;

    fn cblas_saxpy (n : isize, a : libc::c_float, x : *const libc::c_float, incx : isize, y : *mut libc::c_float, incy : isize) -> libc::c_void;
    fn cblas_daxpy (n : isize, a : libc::c_double, x : *const libc::c_double, incx : isize, y : *mut libc::c_double, incy : isize) -> libc::c_void;
    fn cblas_caxpy (n : isize, a : *const c32, x : *const c32, incx : isize, y : *mut c32, incy : isize) -> libc::c_void;
    fn cblas_zaxpy (n : isize, a : *const c64, x : *const c64, incx : isize, y : *mut c64, incy : isize) -> libc::c_void;

    fn cblas_sdot (n : isize, x : *const libc::c_float, incx : isize, y : *const libc::c_float, incy : isize) -> libc::c_float;
    fn cblas_ddot (n : isize, x : *const libc::c_double, incx : isize, y : *const libc::c_double, incy : isize) -> libc::c_double;
    fn cblas_cdotu_sub (n : isize, x : *const c32, incx : isize, y : *const c32, incy : isize, dotu : *mut c32) -> libc::c_void;
    fn cblas_zdotu_sub (n : isize, x : *const c64, incx : isize, y : *const c64, incy : isize, dotu : *mut c64) -> libc::c_void;

    fn cblas_sgemv (layout : libc::c_int, trans : libc::c_int, m : isize, n : isize, alpha : libc::c_float, a : *const libc::c_float, lda : isize, x : *const libc::c_float, incx : isize, beta :libc::c_float, y : *mut libc::c_float, incy : isize) -> libc::c_void;
    fn cblas_dgemv (layout : libc::c_int, trans : libc::c_int, m : isize, n : isize, alpha : libc::c_double, a : *const libc::c_double, lda : isize, x : *const libc::c_double, incx : isize, beta :libc::c_double, y : *mut libc::c_double, incy : isize) -> libc::c_void;
    fn cblas_cgemv (layout : libc::c_int, trans : libc::c_int, m : isize, n : isize, alpha : *const c32, a : *const c32, lda : isize, x : *const c32, incx : isize, beta : *const c32, y : *mut c32, incy : isize) -> libc::c_void;
    fn cblas_zgemv (layout : libc::c_int, trans : libc::c_int, m : isize, n : isize, alpha : *const c64, a : *const c64, lda : isize, x : *const c64, incx : isize, beta : *const c64, y : *mut c64, incy : isize) -> libc::c_void;

    fn cblas_sgemm (layout : libc::c_int, transa : libc::c_int, transb : libc::c_int, m : isize, n : isize, k : isize, alpha : libc::c_float, a : *const libc::c_float, lda : isize, b : *const libc::c_float, ldb : isize, beta : libc::c_float, c : *mut libc::c_float, ldc : isize) -> libc::c_void;
    fn cblas_dgemm (layout : libc::c_int, transa : libc::c_int, transb : libc::c_int, m : isize, n : isize, k : isize, alpha : libc::c_double, a : *const libc::c_double, lda : isize, b : *const libc::c_double, ldb : isize, beta : libc::c_double, c : *mut libc::c_double, ldc : isize) -> libc::c_void;
    fn cblas_cgemm (layout : libc::c_int, transa : libc::c_int, transb : libc::c_int, m : isize, n : isize, k : isize, alpha : *const c32, a : *const c32, lda : isize, b : *const c32, ldb : isize, beta : *const c32, c : *mut c32, ldc : isize) -> libc::c_void;
    fn cblas_zgemm (layout : libc::c_int, transa : libc::c_int, transb : libc::c_int, m : isize, n : isize, k : isize, alpha : *const c64, a : *const c64, lda : isize, b : *const c64, ldb : isize, beta : *const c64, c : *mut c64, ldc : isize) -> libc::c_void;
}

/// A trait representing N-dimensional array on which blas functions can be applied.
pub trait Blas<T : Clone + Display> : NDDataMut<T> {

    /// Compute an element-wise sum of the N-dimensional array.
    fn asum(&self) -> T;

    /// Compute the euclidian norm of the N-dimensional array.
    fn nrm2(&self) -> T;

    /// Scale the N-dimensional array by a factor a.
    fn scal(&mut self, a : T);

    /// Compute y += a * x where y is this array, x is a N-dimensional array of the same size as y 
    /// and a is a scalar.
    fn axpy(&mut self, a : T, x : &NDData<T>);

    /// Compute the dot product of this array and x.
    fn dot(&self, x : &NDData<T>) -> T;

    /// Compute y = alpha * a * x + beta * y where y is this array, x is a one dimensional array, a 
    /// a two dimensional array and alpha and beta are scalars.
    /// Automatically determine whether a need to be transposed.
    fn gemv(&mut self, alpha : T, a : &NDData<T>, x : &NDData<T>, beta : T);

    /// Compute y = alpha * a * x + beta * y where y is this array, a and b are two dimensional 
    /// arrays and alpha and beta are scalars.
    /// Automatically determine whether a and b need to be transposed.
    fn gemm(&mut self, alpha : T, a : &NDData<T>, b : &NDData<T>, beta : T);
}

fn get_mv_trans(m : usize, n : usize, xs : usize, ys : usize) -> libc::c_int {
    if m == ys && n == xs {
        return CBLAS_NO_TRANS;
    }
    else if m == xs && n == ys {
        return CBLAS_TRANS;
    }
    return 0;
}

fn get_mm_trans(am : usize, an : usize, bm : usize, bn : usize, cm : usize, cn : usize) -> (libc::c_int,libc::c_int,usize) {
    if cm == am && cn == bn && an == bm {
        return (CBLAS_NO_TRANS, CBLAS_NO_TRANS, an);
    }
    else if cm == an && cn == bn && am == bm {
        return (CBLAS_TRANS, CBLAS_NO_TRANS, am);
    }
    else if cm == am && cn == bm && an == bn {
        return (CBLAS_NO_TRANS, CBLAS_TRANS, an);
    }
    else if cm == an && cn == bm && am == bn {
        return (CBLAS_TRANS, CBLAS_TRANS, am);
    }
    return (0,0,0);
}

impl<R> Blas<f32> for R where R : NDDataMut<f32> {

    fn asum(&self) -> f32 {
        unsafe {
            cblas_sasum(self.size() as isize, self.get_data().as_ptr(), 1) as f32
        }
    }

    fn nrm2(&self) -> f32 {
        unsafe {
            cblas_snrm2(self.size() as isize, self.get_data().as_ptr(), 1) as f32
        }
    }

    fn scal(&mut self, a : f32) {
        unsafe {
            cblas_sscal(self.size() as isize, a, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    fn axpy(&mut self, a : f32, x : &NDData<f32>) {
        if self.dim() != x.dim() {
            panic!("Blas::axpy(): x is not of the same dimension ({} != {})",  x.dim(), self.dim());
        }

        if self.shape() != x.shape() {
            panic!("Blas::axpy(): x is not of the same shape ({:?} != {:?})",  x.shape(), self.shape());
        }

        unsafe {
            cblas_saxpy(self.size() as isize, a, x.get_data().as_ptr(), 1, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    fn dot(&self, x : &NDData<f32>) -> f32 {
        if self.dim() != 1 || x.dim() != 1{
            panic!("Blas::dot(): x or self is not in 1 dimension ({} != 1 || {} != 1)",  x.dim(), self.dim());
        }
        unsafe {
            cblas_sdot(self.size() as isize, x.get_data().as_ptr(), 1, self.get_data().as_ptr(), 1) as f32
        }
    }

    #[allow(unused_assignments)]
    fn gemv(&mut self, alpha : f32, a : &NDData<f32>, x : &NDData<f32>, beta : f32) {
        if self.dim() != 1 || a.dim() != 2 {
            panic!("Blas::gemv(): self is not in 1 dimension or a is not in 2 dimension({} != 1 || {} != 2)", self.dim(), a.dim());
        }

        let trans = get_mv_trans(a.shape()[0], a.shape()[1], x.shape()[0], self.shape()[0]);
        if trans == 0 {
            panic!("Blas::gemv(): a dimensions ({:?} do not match y ({:?}) and x ({:?})", a.shape(), self.shape(), x.shape());
        }

        unsafe {
            cblas_sgemv(CBLAS_ROW_MAJOR, trans, a.shape()[0] as isize, a.shape()[1] as isize, alpha, a.get_data().as_ptr(), a.strides()[0] as isize, x.get_data().as_ptr(), 1, beta, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    #[allow(unused_assignments)]
    fn gemm(&mut self, alpha : f32, a : &NDData<f32>, b : &NDData<f32>, beta : f32) {
        if self.dim() != 2 || a.dim() != 2 || b.dim() != 2 {
            panic!("Blas::gemm(): a, b or c is not in 2 dimension({} != 2 || {} != 2 || {} != 2)", a.dim(), b.dim(), self.dim());
        }

        let (transa, transb, k) = get_mm_trans(a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1], self.shape()[0], self.shape()[1]);
        if transa == 0 {
            panic!("Blas::gemm(): a dimensions ({:?} do not match b ({:?}) and c ({:?})", a.shape(), b.shape(), self.shape());
        }

        unsafe {
            cblas_sgemm(CBLAS_ROW_MAJOR, transa, transb, self.shape()[0] as isize, self.shape()[1] as isize, k as isize, alpha, a.get_data().as_ptr(), a.strides()[0] as isize, b.get_data().as_ptr(), b.strides()[0] as isize, beta, self.get_data_mut().as_mut_ptr(), self.strides()[0] as isize);
        }
    }
}

impl<R> Blas<f64> for R where R : NDDataMut<f64> {

    fn asum(&self) -> f64 {
        unsafe {
            cblas_dasum(self.size() as isize, self.get_data().as_ptr(), 1) as f64
        }
    }

    fn nrm2(&self) -> f64 {
        unsafe {
            cblas_dnrm2(self.size() as isize, self.get_data().as_ptr(), 1) as f64
        }
    }

    fn scal(&mut self, a : f64) {
        unsafe {
            cblas_dscal(self.size() as isize, a, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    fn axpy(&mut self, a : f64, x : &NDData<f64>) {
        if self.dim() != x.dim() {
            panic!("Blas::axpy(): x is not of the same dimension ({} != {})",  x.dim(), self.dim());
        }

        if self.shape() != x.shape() {
            panic!("Blas::axpy(): x is not of the same shape ({:?} != {:?})",  x.shape(), self.shape());
        }

        unsafe {
            cblas_daxpy(self.size() as isize, a, x.get_data().as_ptr(), 1, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    fn dot(&self, x : &NDData<f64>) -> f64 {
        if self.dim() != 1 || x.dim() != 1{
            panic!("Blas::dot(): x or self is not in 1 dimension ({} != 1 || {} != 1)",  x.dim(), self.dim());
        }
        unsafe {
            cblas_ddot(self.size() as isize, x.get_data().as_ptr(), 1, self.get_data().as_ptr(), 1) as f64
        }
    }

    #[allow(unused_assignments)]
    fn gemv(&mut self, alpha : f64, a : &NDData<f64>, x : &NDData<f64>, beta : f64) {
        if self.dim() != 1 || a.dim() != 2 {
            panic!("Blas::gemv(): self is not in 1 dimension or a is not in 2 dimension({} != 1 || {} != 2)", self.dim(), x.dim());
        }

        let trans = get_mv_trans(a.shape()[0], a.shape()[1], x.shape()[0], self.shape()[0]);
        if trans == 0 {
            panic!("Blas::gemv(): a dimensions ({:?} do not match y ({:?}) and x ({:?})", a.shape(), self.shape(), x.shape());
        }

        unsafe {
            cblas_dgemv(CBLAS_ROW_MAJOR, trans, a.shape()[0] as isize, a.shape()[1] as isize, alpha, a.get_data().as_ptr(), a.strides()[0] as isize, x.get_data().as_ptr(), 1, beta, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    #[allow(unused_assignments)]
    fn gemm(&mut self, alpha : f64, a : &NDData<f64>, b : &NDData<f64>, beta : f64) {
        if self.dim() != 2 || a.dim() != 2 || b.dim() != 2 {
            panic!("Blas::gemm(): a, b or c is not in 2 dimension({} != 2 || {} != 2 || {} != 2)", a.dim(), b.dim(), self.dim());
        }

        let (transa, transb, k) = get_mm_trans(a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1], self.shape()[0], self.shape()[1]);
        if transa == 0 {
            panic!("Blas::gemm(): a dimensions ({:?} do not match b ({:?}) and c ({:?})", a.shape(), b.shape(), self.shape());
        }

        unsafe {
            cblas_dgemm(CBLAS_ROW_MAJOR, transa, transb, self.shape()[0] as isize, self.shape()[1] as isize, k as isize, alpha, a.get_data().as_ptr(), a.strides()[0] as isize, b.get_data().as_ptr(), b.strides()[0] as isize, beta, self.get_data_mut().as_mut_ptr(), self.strides()[0] as isize);
        }
    }
}

impl<R> Blas<c32> for R where R : NDDataMut<c32> {

    fn asum(&self) -> c32 {
        unsafe {
            c32::new(cblas_scasum(self.size() as isize, self.get_data().as_ptr(), 1),0.0)
        }
    }

    fn nrm2(&self) -> c32 {
        unsafe {
            c32::new(cblas_scnrm2(self.size() as isize, self.get_data().as_ptr(), 1),0.0)
        }
    }

    fn scal(&mut self, a : c32) {
        unsafe {
            cblas_cscal(self.size() as isize, &a, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    fn axpy(&mut self, a : c32, x : &NDData<c32>) {
        if self.dim() != x.dim() {
            panic!("Blas::axpy(): x is not of the same dimension ({} != {})",  x.dim(), self.dim());
        }

        if self.shape() != x.shape() {
            panic!("Blas::axpy(): x is not of the same shape ({:?} != {:?})",  x.shape(), self.shape());
        }

        unsafe {
            cblas_caxpy(self.size() as isize, &a, x.get_data().as_ptr(), 1, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    fn dot(&self, x : &NDData<c32>) -> c32 {
        if self.dim() != 1 || x.dim() != 1{
            panic!("Blas::dot(): x or self is not in 1 dimension ({} != 1 || {} != 1)",  x.dim(), self.dim());
        }
        let mut dotu = c32::new(0.0,0.0);
        unsafe {
            cblas_cdotu_sub(self.size() as isize, x.get_data().as_ptr(), 1, self.get_data().as_ptr(), 1, &mut dotu);
        }
        return dotu;
    }

    #[allow(unused_assignments)]
    fn gemv(&mut self, alpha : c32, a : &NDData<c32>, x : &NDData<c32>, beta : c32) {
        if self.dim() != 1 || a.dim() != 2 {
            panic!("Blas::gemv(): self is not in 1 dimension or a is not in 2 dimension({} != 1 || {} != 2)", self.dim(), x.dim());
        }

        let trans = get_mv_trans(a.shape()[0], a.shape()[1], x.shape()[0], self.shape()[0]);
        if trans == 0 {
            panic!("Blas::gemv(): a dimensions ({:?} do not match y ({:?}) and x ({:?})", a.shape(), self.shape(), x.shape());
        }

        unsafe {
            cblas_cgemv(CBLAS_ROW_MAJOR, trans, a.shape()[0] as isize, a.shape()[1] as isize, &alpha, a.get_data().as_ptr(), a.strides()[0] as isize, x.get_data().as_ptr(), 1, &beta, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    #[allow(unused_assignments)]
    fn gemm(&mut self, alpha : c32, a : &NDData<c32>, b : &NDData<c32>, beta : c32) {
        if self.dim() != 2 || a.dim() != 2 || b.dim() != 2 {
            panic!("Blas::gemm(): a, b or c is not in 2 dimension({} != 2 || {} != 2 || {} != 2)", a.dim(), b.dim(), self.dim());
        }

        let (transa, transb, k) = get_mm_trans(a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1], self.shape()[0], self.shape()[1]);
        if transa == 0 {
            panic!("Blas::gemm(): a dimensions ({:?} do not match b ({:?}) and c ({:?})", a.shape(), b.shape(), self.shape());
        }

        unsafe {
            cblas_cgemm(CBLAS_ROW_MAJOR, transa, transb, self.shape()[0] as isize, self.shape()[1] as isize, k as isize, &alpha, a.get_data().as_ptr(), a.strides()[0] as isize, b.get_data().as_ptr(), b.strides()[0] as isize, &beta, self.get_data_mut().as_mut_ptr(), self.strides()[0] as isize);
        }
    }
}

impl<R> Blas<c64> for R where R : NDDataMut<c64> {

    fn asum(&self) -> c64 {
        unsafe {
            c64::new(cblas_dzasum(self.size() as isize, self.get_data().as_ptr(), 1),0.0)
        }
    }

    fn nrm2(&self) -> c64 {
        unsafe {
            c64::new(cblas_dznrm2(self.size() as isize, self.get_data().as_ptr(), 1),0.0)
        }
    }

    fn scal(&mut self, a : c64) {
        unsafe {
            cblas_zscal(self.size() as isize, &a, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    fn axpy(&mut self, a : c64, x : &NDData<c64>) {
        if self.dim() != x.dim() {
            panic!("Blas::axpy(): x is not of the same dimension ({} != {})",  x.dim(), self.dim());
        }

        if self.shape() != x.shape() {
            panic!("Blas::axpy(): x is not of the same shape ({:?} != {:?})",  x.shape(), self.shape());
        }

        unsafe {
            cblas_zaxpy(self.size() as isize, &a, x.get_data().as_ptr(), 1, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    fn dot(&self, x : &NDData<c64>) -> c64 {
        if self.dim() != 1 || x.dim() != 1{
            panic!("Blas::dot(): x or self is not in 1 dimension ({} != 1 || {} != 1)",  x.dim(), self.dim());
        }
        let mut dotu = c64::new(0.0,0.0);
        unsafe {
            cblas_zdotu_sub(self.size() as isize, x.get_data().as_ptr(), 1, self.get_data().as_ptr(), 1, &mut dotu);
        }
        return dotu;
    }

    #[allow(unused_assignments)]
    fn gemv(&mut self, alpha : c64, a : &NDData<c64>, x : &NDData<c64>, beta : c64) {
        if self.dim() != 1 || a.dim() != 2 {
            panic!("Blas::gemv(): self is not in 1 dimension or a is not in 2 dimension({} != 1 || {} != 2)", self.dim(), x.dim());
        }

        let trans = get_mv_trans(a.shape()[0], a.shape()[1], x.shape()[0], self.shape()[0]);
        if trans == 0 {
            panic!("Blas::gemv(): a dimensions ({:?} do not match y ({:?}) and x ({:?})", a.shape(), self.shape(), x.shape());
        }

        unsafe {
            cblas_zgemv(CBLAS_ROW_MAJOR, trans, a.shape()[0] as isize, a.shape()[1] as isize, &alpha, a.get_data().as_ptr(), a.strides()[0] as isize, x.get_data().as_ptr(), 1, &beta, self.get_data_mut().as_mut_ptr(), 1);
        }
    }

    #[allow(unused_assignments)]
    fn gemm(&mut self, alpha : c64, a : &NDData<c64>, b : &NDData<c64>, beta : c64) {
        if self.dim() != 2 || a.dim() != 2 || b.dim() != 2 {
            panic!("Blas::gemm(): a, b or c is not in 2 dimension({} != 2 || {} != 2 || {} != 2)", a.dim(), b.dim(), self.dim());
        }

        let (transa, transb, k) = get_mm_trans(a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1], self.shape()[0], self.shape()[1]);
        if transa == 0 {
            panic!("Blas::gemm(): a dimensions ({:?} do not match b ({:?}) and c ({:?})", a.shape(), b.shape(), self.shape());
        }

        unsafe {
            cblas_zgemm(CBLAS_ROW_MAJOR, transa, transb, self.shape()[0] as isize, self.shape()[1] as isize, k as isize, &alpha, a.get_data().as_ptr(), a.strides()[0] as isize, b.get_data().as_ptr(), b.strides()[0] as isize, &beta, self.get_data_mut().as_mut_ptr(), self.strides()[0] as isize);
        }
    }
}

/*
==================== AddAssign ====================
*/

impl<'a, T : Clone + Display + From<f32>, R : NDData<T> + Sized> AddAssign<&'a R> for NDArray<T> where NDArray<T> : Blas<T> {

    fn add_assign(&mut self, rhs: &'a R) {
        self.axpy(T::from(1.0f32), rhs);
    }
}

impl<'a, 'b, T : Clone + Display + From<f32>, R : NDData<T> + Sized> AddAssign<&'a R> for NDSliceMut<'b, T> where NDSliceMut<'b, T> : Blas<T> {

    fn add_assign(&mut self, rhs: &'a R) {
        self.axpy(T::from(1.0f32), rhs);
    }
}

/*
==================== Add ====================
*/

impl<'a, 'b, T : Clone + Display + From<f32>, R : NDData<T> + Sized> Add<&'a R> for &'b NDArray<T> where NDArray<T> : Blas<T> {
    type Output = NDArray<T>;

    fn add(self, rhs: &'a R) -> NDArray<T> {
        let mut res = NDArray::<T>::copy(self);
        res += rhs;
        return res;
    }
}

impl<'a, 'b, 'c, T : Clone + Display + From<f32>, R : NDData<T> + Sized> Add<&'a R> for &'b NDSlice<'c, T> where NDArray<T> : Blas<T> {
    type Output = NDArray<T>;

    fn add(self, rhs: &'a R) -> NDArray<T> {
        let mut res = NDArray::<T>::copy(self);
        res += rhs;
        return res;
    }
}

impl<'a, 'b, 'c, T : Clone + Display + From<f32>, R : NDData<T> + Sized> Add<&'a R> for &'b NDSliceMut<'c, T> where NDArray<T> : Blas<T> {
    type Output = NDArray<T>;

    fn add(self, rhs: &'a R) -> NDArray<T> {
        let mut res = NDArray::<T>::copy(self);
        res += rhs;
        return res;
    }
}

/*
==================== SubAssign ====================
*/

impl<'a, T : Clone + Display + From<f32>, R : NDData<T> + Sized> SubAssign<&'a R> for NDArray<T> where NDArray<T> : Blas<T> {

    fn sub_assign(&mut self, rhs: &'a R) {
        self.axpy(T::from(-1.0f32), rhs);
    }
}

impl<'a, 'b, T : Clone + Display + From<f32>, R : NDData<T> + Sized> SubAssign<&'a R> for NDSliceMut<'b, T> where NDSliceMut<'b, T> : Blas<T> {

    fn sub_assign(&mut self, rhs: &'a R) {
        self.axpy(T::from(-1.0f32), rhs);
    }
}

/*
==================== Sub ====================
*/

impl<'a, 'b, T : Clone + Display + From<f32>, R : NDData<T> + Sized> Sub<&'a R> for &'b NDArray<T> where NDArray<T> : Blas<T> {
    type Output = NDArray<T>;

    fn sub(self, rhs: &'a R) -> NDArray<T> {
        let mut res = NDArray::<T>::copy(self);
        res -= rhs;
        return res;
    }
}

impl<'a, 'b, 'c, T : Clone + Display + From<f32>, R : NDData<T> + Sized> Sub<&'a R> for &'b NDSlice<'c, T> where NDArray<T> : Blas<T> {
    type Output = NDArray<T>;

    fn sub(self, rhs: &'a R) -> NDArray<T> {
        let mut res = NDArray::<T>::copy(self);
        res -= rhs;
        return res;
    }
}


impl<'a, 'b, 'c, T : Clone + Display + From<f32>, R : NDData<T> + Sized> Sub<&'a R> for &'b NDSliceMut<'c, T> where NDArray<T> : Blas<T> {
    type Output = NDArray<T>;

    fn sub(self, rhs: &'a R) -> NDArray<T> {
        let mut res = NDArray::<T>::copy(self);
        res -= rhs;
        return res;
    }
}

/*
==================== MulAssign ====================
*/

impl<'a, T : Clone + Display + From<f32>, I : NDData<T> + Sized> MulAssign<&'a I> for NDArray<T> where NDArray<T> : Blas<T> + NDData<T> {

    fn mul_assign(&mut self, rhs: &'a I) {
        if self.dim() == 1 && rhs.dim() == 2 {
            let x = NDArray::<T>::copy(self);
            self.gemv(T::from(1.0), rhs, &x, T::from(0.0));
        }
        else if self.dim() == 2 && rhs.dim() == 2 {
            let a = NDArray::<T>::copy(self);
            self.gemm(T::from(1.0), &a, rhs, T::from(0.0));
        }
        else {
            panic!("MulAssign could not find a blas operation between NDData of dimension {} and {}", self.dim(), rhs.dim());
        }
    }
}

impl<'a, 'b, T : Clone + Display + From<f32>, I : NDData<T> + Sized> MulAssign<&'b I> for NDSliceMut<'a, T> where NDSliceMut<'a, T> : Blas<T> + NDData<T> {

    fn mul_assign(&mut self, rhs: &'b I) {
        if self.dim() == 1 && rhs.dim() == 2 {
            let x = NDArray::<T>::copy(self);
            self.gemv(T::from(1.0), rhs, &x, T::from(0.0));
        }
        else if self.dim() == 2 && rhs.dim() == 2 {
            let a = NDArray::<T>::copy(self);
            self.gemm(T::from(1.0), &a, rhs, T::from(0.0));
        }
        else {
            panic!("MulAssign could not find a blas operation between NDData of dimension {} and {}", self.dim(), rhs.dim());
        }
    }
}

/*
==================== Mul ====================
*/

impl<'a, 'b, T : Clone + Display + From<f32>, R : NDData<T> + Sized> Mul<&'a R> for &'b NDArray<T> where NDArray<T> : Blas<T> {
    type Output = NDArray<T>;

    fn mul(self, rhs: &'a R) -> NDArray<T> {
        let mut res = NDArray::<T>::copy(self);
        res *= rhs;
        return res;
    }
}

impl<'a, 'b, 'c, T : Clone + Display + From<f32>, R : NDData<T> + Sized> Mul<&'a R> for &'b NDSlice<'c, T> where NDArray<T> : Blas<T> {
    type Output = NDArray<T>;

    fn mul(self, rhs: &'a R) -> NDArray<T> {
        let mut res = NDArray::<T>::copy(self);
        res *= rhs;
        return res;
    }
}
impl<'a, 'b, 'c, T : Clone + Display + From<f32>, R : NDData<T> + Sized> Mul<&'a R> for &'b NDSliceMut<'c, T> where NDArray<T> : Blas<T> {
    type Output = NDArray<T>;

    fn mul(self, rhs: &'a R) -> NDArray<T> {
        let mut res = NDArray::<T>::copy(self);
        res *= rhs;
        return res;
    }
}
