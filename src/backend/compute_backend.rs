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

pub trait ComputeBackend {

    fn init(&mut self) -> Result<(),String>;

    fn create_array(&mut self, size : usize) -> Result<u32,String>;

    fn set_array(&self, id : u32, array : &SizedBuffer) -> Result<(),String>;

    fn get_array(&self, id : u32, array : &mut SizedBufferMut) -> Result<(),String>;

    fn delete_array(&mut self, id : u32) -> Result<(),String>;

    fn finalize(&mut self) -> Result<(),String>;
}

pub struct ArrayRegistry<T : Clone> {
    array_list : Vec<T>,
    array_size : Vec<usize>,
    free_list : Vec<usize>,
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

impl<T : Sized + Clone> ArrayRegistry<T> {
    pub fn new() -> ArrayRegistry<T> {
        ArrayRegistry::<T> {
            array_list : Vec::<T>::new(),
            array_size : Vec::<usize>::new(),
            free_list : Vec::<usize>::new(),
        }
    }

    pub fn register_array(&mut self, array : T, size : usize) -> u32 {
        if size == 0 {
            panic!("Attempting to register an emtpy array");
        }

        if let Some(idx) = self.free_list.pop() {
            self.array_list[idx] = array.clone();
            return idx as u32;
        }
        else {
            self.array_list.push(array.clone());
            self.array_size.push(size);
            return (self.array_list.len() - 1) as u32;
        }
    }

    pub fn get_array(&self, id : u32) -> T {
        if id as usize >= self.array_list.len() {
            panic!("Querying array id {} which is not in the ArrayRegistry", id);
        }
        if self.array_size[id as usize] == 0 {
            panic!("Querying array id {} which has already been unregistered", id);
        }
        return self.array_list[id as usize].clone();
    }

    pub fn get_array_size(&self, id : u32) -> usize {
        if id as usize >= self.array_list.len() {
            panic!("Querying array id {} which is not in the ArrayRegistry", id);
        }
        return self.array_size[id as usize];
    }

    pub fn unregister_array(&mut self, id : u32) {
        if id as usize >= self.array_list.len() {
            panic!("Unregistering array id {} which is not in the ArrayRegistry", id);
        }
        self.array_size[id as usize] = 0;
        self.free_list.push(id as usize);
    }
}
