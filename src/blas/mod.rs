extern crate num;
extern crate libc;

use std::fmt::Display;
use std::ops::{AddAssign, SubAssign, MulAssign};

use array::{NDData, NDDataMut, NDArray, NDSliceMut};

const CBLAS_ROW_MAJOR : libc::c_int = 101;
#[allow(dead_code)]
const CBLAS_COL_MAJOR : libc::c_int = 102;
const CBLAS_NO_TRANS : libc::c_int = 111;
#[allow(dead_code)]
const CBLAS_TRANS : libc::c_int = 112;

#[link(name = "blas")]
extern {
    fn cblas_sasum (n : isize, x : *const libc::c_float, incx : isize) -> libc::c_float;
    fn cblas_dasum (n : isize, x : *const libc::c_double, incx : isize) -> libc::c_double;
    fn cblas_snrm2 (n : isize, x : *const libc::c_float, incx : isize) -> libc::c_float;
    fn cblas_dnrm2 (n : isize, x : *const libc::c_double, incx : isize) -> libc::c_double;
    fn cblas_sscal (n : isize, a : libc::c_float, x : *mut libc::c_float, incx : isize) -> libc::c_void;
    fn cblas_dscal (n : isize, a : libc::c_double, x : *mut libc::c_double, incx : isize) -> libc::c_void;
    fn cblas_saxpy (n : isize, a : libc::c_float, x : *const libc::c_float, incx : isize, y : *mut libc::c_float, incy : isize) -> libc::c_void;
    fn cblas_daxpy (n : isize, a : libc::c_double, x : *const libc::c_double, incx : isize, y : *mut libc::c_double, incy : isize) -> libc::c_void;
    fn cblas_sdot (n : isize, x : *const libc::c_float, incx : isize, y : *const libc::c_float, incy : isize) -> libc::c_float;
    fn cblas_ddot (n : isize, x : *const libc::c_double, incx : isize, y : *const libc::c_double, incy : isize) -> libc::c_double;
    fn cblas_sgemv (layout : libc::c_int, trans : libc::c_int, m : isize, n : isize, alpha : libc::c_float, a : *const libc::c_float, lda : isize, x : *const libc::c_float, incx : isize, beta :libc::c_float, y : *mut libc::c_float, incy : isize) -> libc::c_void;
    fn cblas_dgemv (layout : libc::c_int, trans : libc::c_int, m : isize, n : isize, alpha : libc::c_double, a : *const libc::c_double, lda : isize, x : *const libc::c_double, incx : isize, beta :libc::c_double, y : *mut libc::c_double, incy : isize) -> libc::c_void;
    fn cblas_sgemm (layout : libc::c_int, transa : libc::c_int, transb : libc::c_int, m : isize, n : isize, k : isize, alpha : libc::c_float, a : *const libc::c_float, lda : isize, b : *const libc::c_float, ldb : isize, beta : libc::c_float, c : *mut libc::c_float, ldc : isize) -> libc::c_void;
    fn cblas_dgemm (layout : libc::c_int, transa : libc::c_int, transb : libc::c_int, m : isize, n : isize, k : isize, alpha : libc::c_double, a : *const libc::c_double, lda : isize, b : *const libc::c_double, ldb : isize, beta : libc::c_double, c : *mut libc::c_double, ldc : isize) -> libc::c_void;
}

pub trait Blas<T : Clone + Display> : NDDataMut<T> {

    fn asum(&self) -> T;

    fn nrm2(&self) -> T;

    fn scal(&mut self, a : T);

    fn axpy(&mut self, a : T, x : &NDData<T>);

    fn dot(&self, x : &NDData<T>) -> T;

    fn gemv(&mut self, alpha : T, a : &NDData<T>, x : &NDData<T>, beta : T);

    fn gemm(&mut self, alpha : T, a : &NDData<T>, b : &NDData<T>, beta : T);
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

        let mut trans = 0;
        if a.shape()[0] == self.shape()[0] && a.shape()[1] == x.shape()[0] {
            trans = CBLAS_NO_TRANS
        }
        else if  a.shape()[0] == x.shape()[0] && a.shape()[1] == self.shape()[0] {
            trans = CBLAS_TRANS
        }
        else {
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

        let mut transa = 0;
        let mut transb = 0;
        let mut k = 0;
        if self.shape()[0] == a.shape()[0] && self.shape()[1] == b.shape()[1] && a.shape()[1] == b.shape()[0] {
            transa = CBLAS_NO_TRANS;
            transb = CBLAS_NO_TRANS;
            k = a.shape()[1];
        }
        else if self.shape()[0] == a.shape()[1] && self.shape()[1] == b.shape()[1] && a.shape()[0] == b.shape()[0] {
            transa = CBLAS_TRANS;
            transb = CBLAS_NO_TRANS;
            k = a.shape()[0];
        }
        else if self.shape()[0] == a.shape()[0] && self.shape()[1] == b.shape()[0] && a.shape()[1] == b.shape()[1] {
            transa = CBLAS_NO_TRANS;
            transb = CBLAS_TRANS;
            k = a.shape()[1];
        }
        else if self.shape()[0] == a.shape()[1] && self.shape()[1] == b.shape()[0] && a.shape()[0] == b.shape()[1] {
            transa = CBLAS_TRANS;
            transb = CBLAS_TRANS;
            k = a.shape()[0];
        }
        else {
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
        let mut trans = 0;
        if a.shape()[0] == self.shape()[0] && a.shape()[1] == x.shape()[0] {
            trans = CBLAS_NO_TRANS
        }
        else if  a.shape()[0] == x.shape()[0] && a.shape()[1] == self.shape()[0] {
            trans = CBLAS_TRANS
        }
        else {
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

        let mut transa = 0;
        let mut transb = 0;
        let mut k = 0;
        if self.shape()[0] == a.shape()[0] && self.shape()[1] == b.shape()[1] && a.shape()[1] == b.shape()[0] {
            transa = CBLAS_NO_TRANS;
            transb = CBLAS_NO_TRANS;
            k = a.shape()[1];
        }
        else if self.shape()[0] == a.shape()[1] && self.shape()[1] == b.shape()[1] && a.shape()[0] == b.shape()[0] {
            transa = CBLAS_TRANS;
            transb = CBLAS_NO_TRANS;
            k = a.shape()[0];
        }
        else if self.shape()[0] == a.shape()[0] && self.shape()[1] == b.shape()[0] && a.shape()[1] == b.shape()[1] {
            transa = CBLAS_NO_TRANS;
            transb = CBLAS_TRANS;
            k = a.shape()[1];
        }
        else if self.shape()[0] == a.shape()[1] && self.shape()[1] == b.shape()[0] && a.shape()[0] == b.shape()[1] {
            transa = CBLAS_TRANS;
            transb = CBLAS_TRANS;
            k = a.shape()[0];
        }
        else {
            panic!("Blas::gemm(): a dimensions ({:?} do not match b ({:?}) and c ({:?})", a.shape(), b.shape(), self.shape());
        }
        unsafe {
            cblas_dgemm(CBLAS_ROW_MAJOR, transa, transb, self.shape()[0] as isize, self.shape()[1] as isize, k as isize, alpha, a.get_data().as_ptr(), a.strides()[0] as isize, b.get_data().as_ptr(), b.strides()[0] as isize, beta, self.get_data_mut().as_mut_ptr(), self.strides()[0] as isize);
        }
    }
}

/*
==================== AddAssign ====================
*/

impl<T : Clone + Display + From<f32>, R : NDData<T> + Sized> AddAssign<R> for NDArray<T> where NDArray<T> : Blas<T> {

    #[inline(always)]
    fn add_assign(&mut self, rhs: R) {
        self.axpy(T::from(1.0f32), &rhs);
    }
}

impl<'b, T : Clone + Display + From<f32>, R : NDData<T> + Sized> AddAssign<R> for NDSliceMut<'b, T> where NDSliceMut<'b, T> : Blas<T> {

    #[inline(always)]
    fn add_assign(&mut self, rhs: R) {
        self.axpy(T::from(1.0f32), &rhs);
    }
}

/*
==================== SubAssign ====================
*/

impl<T : Clone + Display + From<f32>, R : NDData<T> + Sized> SubAssign<R> for NDArray<T> where NDArray<T> : Blas<T> {

    #[inline(always)]
    fn sub_assign(&mut self, rhs: R) {
        self.axpy(T::from(-1.0f32), &rhs);
    }
}

impl<'b, T : Clone + Display + From<f32>, R : NDData<T> + Sized> SubAssign<R> for NDSliceMut<'b, T> where NDSliceMut<'b, T> : Blas<T> {

    #[inline(always)]
    fn sub_assign(&mut self, rhs: R) {
        self.axpy(T::from(-1.0f32), &rhs);
    }
}

/*
==================== MulAssign ====================
*/

impl<T : Clone + Display + From<f32>, I : NDData<T> + Sized> MulAssign<I> for NDArray<T> where NDArray<T> : Blas<T> + NDData<T> {

    #[inline(always)]
    fn mul_assign(&mut self, rhs: I) {
        if self.dim() == 1 && rhs.dim() == 2 {
            let x = NDArray::<T>::copy(self);
            self.gemv(T::from(1.0), &rhs, &x, T::from(0.0));
        }
        else if self.dim() == 2 && rhs.dim() == 2 {
            let a = NDArray::<T>::copy(self);
            self.gemm(T::from(1.0), &a, &rhs, T::from(0.0));
        }
        else {
            panic!("MulAssign could not find a blas operation between NDData of dimension {} and {}", self.dim(), rhs.dim());
        }
    }
}

impl<'a, T : Clone + Display + From<f32>, I : NDData<T> + Sized> MulAssign<I> for NDSliceMut<'a, T> where NDSliceMut<'a, T> : Blas<T> + NDData<T> {

    #[inline(always)]
    fn mul_assign(&mut self, rhs: I) {
        if self.dim() == 1 && rhs.dim() == 2 {
            let x = NDArray::<T>::copy(self);
            self.gemv(T::from(1.0), &rhs, &x, T::from(0.0));
        }
        else if self.dim() == 2 && rhs.dim() == 2 {
            let a = NDArray::<T>::copy(self);
            self.gemm(T::from(1.0), &a, &rhs, T::from(0.0));
        }
        else {
            panic!("MulAssign could not find a blas operation between NDData of dimension {} and {}", self.dim(), rhs.dim());
        }
    }
}
