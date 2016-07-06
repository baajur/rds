extern crate libc;

use std::fmt::Display;
use std::mem::{transmute, size_of};

use array::{NDData, NDDataMut, NDArray, NDSlice, NDSliceMut};

pub trait SizedBuffer {

    unsafe fn get_raw_ptr(&self) -> *const libc::c_void;

    fn get_raw_size(&self) -> usize;
}

pub trait SizedBufferMut : SizedBuffer {

    unsafe fn get_raw_ptr_mut(&mut self) -> *mut libc::c_void;
}

impl<T> SizedBuffer for NDArray<T> {

    unsafe fn get_raw_ptr(&self) -> *const libc::c_void {
        transmute(self.get_data().as_ptr())
    }

    fn get_raw_size(&self) -> usize {
        self.size() * size_of::<T>()
    }
}

impl<T : Clone + Display> SizedBufferMut for NDArray<T> where NDArray<T> : SizedBuffer {

    unsafe fn get_raw_ptr_mut(&mut self) -> *mut libc::c_void {
        transmute(self.get_data_mut().as_mut_ptr())
    }
}

impl<'a, T> SizedBuffer for NDSliceMut<'a, T> {

    unsafe fn get_raw_ptr(&self) -> *const libc::c_void {
        transmute(self.get_data().as_ptr())
    }

    fn get_raw_size(&self) -> usize {
        self.size() * size_of::<T>()
    }
}

impl<'a, T : Clone + Display> SizedBufferMut for NDSliceMut<'a, T> where NDSliceMut<'a, T> : SizedBuffer {

    unsafe fn get_raw_ptr_mut(&mut self) -> *mut libc::c_void {
        transmute(self.get_data_mut().as_mut_ptr())
    }
}

impl<'a, T> SizedBuffer for NDSlice<'a, T> {

    unsafe fn get_raw_ptr(&self) -> *const libc::c_void {
        transmute(self.get_data().as_ptr())
    }

    fn get_raw_size(&self) -> usize {
        self.size() * size_of::<T>()
    }
}

pub trait ComputeBackend {

    fn init(&mut self) -> Result<(),String>;

    fn create_array() -> u32;
    
    fn set_array(id : u32, array : &SizedBuffer);

    fn get_array(id : u32, array : &mut SizedBufferMut);

    fn delete_array(id : u32);

    fn finalize(&mut self) -> Result<(),String>;
}
