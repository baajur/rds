extern crate num;
extern crate libc;

use std::fmt::Display;

use array::{NDData, NDDataMut};

#[link(name = "blas")]
extern {
    fn cblas_saxpy (n : isize, a : libc::c_float, x : *const libc::c_float, incx : isize, y : *mut libc::c_float, incy : isize) -> libc::c_void;
    fn cblas_daxpy (n : isize, a : libc::c_double, x : *const libc::c_double, incx : isize, y : *mut libc::c_double, incy : isize) -> libc::c_void;
}

pub trait Blas<T : Clone + Display> : NDDataMut<T> {
    fn axpy(&mut self, a : T, x : &NDData<T>);
}

impl<R> Blas<f32> for R where R : NDDataMut<f32> {

    fn axpy(&mut self, a : f32, x : &NDData<f32>) {
        if self.dim() != self.dim() {
            panic!("Blas::cblas_add(): x is not of the same dimension ({} != {})",  x.dim(), self.dim());
        }

        if self.shape() != x.shape() {
            panic!("NDDataMut::cblas_add(): x is not of the same shape ({:?} != {:?})",  x.shape(), self.shape());
        }

        unsafe {
            cblas_saxpy(self.size() as isize, a, x.get_data().as_ptr(), 1, self.get_data_mut().as_mut_ptr(), 1);
        }
    }
}

impl<R> Blas<f64> for R where R : NDDataMut<f64> {

    fn axpy(&mut self, a : f64, x : &NDData<f64>) {
        if self.dim() != self.dim() {
            panic!("Blas::cblas_add(): x is not of the same dimension ({} != {})",  x.dim(), self.dim());
        }

        if self.shape() != x.shape() {
            panic!("NDDataMut::cblas_add(): x is not of the same shape ({:?} != {:?})",  x.shape(), self.shape());
        }

        unsafe {
            cblas_daxpy(self.size() as isize, a, x.get_data().as_ptr(), 1, self.get_data_mut().as_mut_ptr(), 1);
        }
    }
}
