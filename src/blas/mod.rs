extern crate num;
extern crate libc;

use std::fmt::Display;

use array::{NDData, NDDataMut};

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
}

pub trait Blas<T : Clone + Display> : NDDataMut<T> {

    fn asum(&self) -> T;

    fn nrm2(&self) -> T;

    fn scal(&mut self, a : T);

    fn axpy(&mut self, a : T, x : &NDData<T>);

    fn dot(&self, x : &NDData<T>) -> T;
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
}
