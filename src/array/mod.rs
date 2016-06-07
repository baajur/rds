use std::iter::repeat;
use std::ops::{Index, IndexMut};

pub trait NDData<T> {

    fn dim(&self) -> usize;

    fn shape(&self) -> &[usize];

    fn strides(&self) -> &[usize]; 

    fn size(&self) -> usize;

    fn get_data(&self) -> &[T];

    unsafe fn get_data_raw(&self) -> *const T;
}

pub trait NDSliceable<'a, T : 'a> {

    fn slice(&'a self, idx : &[usize]) -> NDSlice<'a, T>;
}

pub trait NDSliceableMut<'a, T : 'a> {

    fn slice_mut(&'a mut self, idx : &[usize]) -> NDSliceMut<'a, T>;
}

pub struct NDSlice<'a, T : 'a> {
    shape : &'a [usize],
    strides : &'a [usize],
    data : &'a [T],
}

pub struct NDSliceMut<'a, T : 'a> {
    shape : &'a [usize],
    strides : &'a [usize],
    data : &'a mut [T],
}

pub struct Array<T> {
    shape : Vec<usize>,
    strides : Vec<usize>,
    data : Box<[T]>,
}

impl<T : Clone> Array<T> {
    pub fn new(shape : &[usize], v : T) -> Array<T> {
        let mut strides : Vec<usize> = repeat(0usize).take(shape.len()).collect();
        let mut size = 1usize;
        for i in 0..shape.len() {
            let revidx = shape.len() - i - 1;
            strides[revidx] = size;
            size *= shape[revidx];
        }
        let alloc : Vec<T> = repeat(v).take(size).collect();

        return Array {
            shape : shape.to_vec(),
            strides : strides,
            data : alloc.into_boxed_slice(),
        }
    }

    pub fn copy<R : NDData<T>>(data : R) -> Array<T> {
        Array {
            shape : data.shape().to_vec(),
            strides : data.strides().to_vec(),
            data : data.get_data().to_vec().into_boxed_slice(),
        }
    }
}


impl<T> NDData<T> for Array<T> { 

    fn dim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape[..]
    }

    fn strides(&self) -> &[usize] {
        &self.strides[..]
    }

    fn size(&self) -> usize {
        self.shape.iter().fold(1usize, |acc, &x| acc * x)
    }

    fn get_data(&self) -> &[T] {
        &self.data[..]
    }

    unsafe fn get_data_raw(&self) -> *const T {
        self.data.as_ptr()
    }
}

impl<'a, T : 'a> NDSliceable<'a, T> for Array<T> {

    fn slice(&'a self, idx : &[usize]) -> NDSlice<'a, T> {
        assert!(idx.len() <= self.shape.len());
        let mut start = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            start += idx[i] * self.strides[i];
        }
        let end = start + self.strides[idx.len()-1];
        NDSlice {
            shape : &self.shape[idx.len()..],
            strides : &self.strides[idx.len()..],
            data : &self.data[start..end]
        }
    }
}

impl<'a, T : 'a> NDSliceableMut<'a, T> for Array<T> {

    fn slice_mut(&'a mut self, idx : &[usize]) -> NDSliceMut<'a, T> {
        assert!(idx.len() <= self.shape.len());
        let mut start = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            start += idx[i] * self.strides[i];
        }
        let end = start + self.strides[idx.len()-1];
        NDSliceMut {
            shape : &self.shape[idx.len()..],
            strides : &self.strides[idx.len()..],
            data : &mut self.data[start..end]
        }
    }
}

impl<'b, T> Index<&'b [usize]> for Array<T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize]) -> &'c T {
        assert!(idx.len() == self.shape.len());
        let mut pos = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            pos += idx[i] * self.strides[i];
        }
        return &self.data[pos];
    }
}

impl<'b, T> IndexMut<&'b [usize]> for Array<T> {

    fn index_mut<'c>(&'c mut self, idx : &[usize]) -> &'c mut T {
        assert!(idx.len() == self.shape.len());
        let mut pos = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            pos += idx[i] * self.strides[i];
        }
        return &mut self.data[pos];
    }
}

impl<'a, T> NDData<T> for NDSlice<'a, T> {

    fn dim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn size(&self) -> usize {
        self.shape.iter().fold(1usize, |acc, &x| acc * x)
    }

    fn get_data(&self) -> &[T] {
        self.data
    }

    unsafe fn get_data_raw(&self) -> *const T {
        self.data.as_ptr()
    }

}

impl<'a, T : 'a> NDSliceable<'a, T> for NDSlice<'a, T> {

    fn slice(&'a self, idx : &[usize]) -> NDSlice<'a, T> {
        assert!(idx.len() <= self.shape.len());
        let mut start = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            start += idx[i] * self.strides[i];
        }
        let end = start + self.strides[idx.len()-1];
        NDSlice {
            shape : &self.shape[idx.len()..],
            strides : &self.strides[idx.len()..],
            data : &self.data[start..end]
        }
    }
}

impl<'a, 'b, T> Index<&'b [usize]> for NDSlice<'a, T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize]) -> &'c T {
        assert!(idx.len() == self.shape.len());
        let mut pos = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            pos += idx[i] * self.strides[i];
        }
        return &self.data[pos];
    }
}

impl<'a, T> NDData<T> for NDSliceMut<'a, T> {

    fn dim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn size(&self) -> usize {
        self.shape.iter().fold(1usize, |acc, &x| acc * x)
    }

    fn get_data(&self) -> &[T] {
        self.data
    }

    unsafe fn get_data_raw(&self) -> *const T {
        self.data.as_ptr()
    }
}

impl<'a, T : 'a> NDSliceable<'a, T> for NDSliceMut<'a, T> {

    fn slice(&'a self, idx : &[usize]) -> NDSlice<'a, T> {
        assert!(idx.len() <= self.shape.len());
        let mut start = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            start += idx[i] * self.strides[i];
        }
        let end = start + self.strides[idx.len()-1];
        NDSlice {
            shape : &self.shape[idx.len()..],
            strides : &self.strides[idx.len()..],
            data : &self.data[start..end]
        }
    }
}

impl<'a, T : 'a> NDSliceableMut<'a, T> for NDSliceMut<'a, T> {

    fn slice_mut(&'a mut self, idx : &[usize]) -> NDSliceMut<'a, T> {
        assert!(idx.len() <= self.shape.len());
        let mut start = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            start += idx[i] * self.strides[i];
        }
        let end = start + self.strides[idx.len()-1];
        NDSliceMut {
            shape : &self.shape[idx.len()..],
            strides : &self.strides[idx.len()..],
            data : &mut self.data[start..end]
        }
    }
}

impl<'a, 'b, T> Index<&'b [usize]> for NDSliceMut<'a, T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize]) -> &'c T {
        assert!(idx.len() == self.shape.len());
        let mut pos = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            pos += idx[i] * self.strides[i];
        }
        return &self.data[pos];
    }
}

impl<'a, 'b, T> IndexMut<&'b [usize]> for NDSliceMut<'a, T> {

    fn index_mut<'c>(&'c mut self, idx : &[usize]) -> &'c mut T {
        assert!(idx.len() == self.shape.len());
        let mut pos = 0usize;
        for i in 0..idx.len() {
            assert!(idx[i] < self.shape[i]);
            pos += idx[i] * self.strides[i];
        }
        return &mut self.data[pos];
    }
}
