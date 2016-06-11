use std::cmp::{PartialEq, Eq};
use std::iter::repeat;
use std::fmt::Display;
use std::ops::{Index, IndexMut};

pub mod csv;
pub mod numpy;

pub trait NDData<T> {

    fn shape(&self) -> &[usize];

    fn strides(&self) -> &[usize]; 

    fn get_data(&self) -> &[T];

    fn dim(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().iter().fold(1usize, |acc, &x| acc * x)
    }

    fn idx<'a>(&'a self, idx : &[usize]) -> &'a T {
        if idx.len() != self.shape().len() {
            panic!("NDData::idx({:?}): idx is not of the right dimension ({} != {})", idx, idx.len(), self.dim());
        }
        let mut pos = 0usize;
        for i in 0..idx.len() {
            if idx[i] >= self.shape()[i] {
                panic!("NDData::idx({:?}): idx is out of bound for dimension {} (shape: {:?})", idx, i, self.shape());
            }
            pos += idx[i] * self.strides()[i];
        }
        return &self.get_data()[pos];
    }
}

pub trait NDDataMut<T : Clone + Display> : NDData<T> {

    fn get_data_mut(&mut self) -> &mut [T];

    fn idx_mut<'a>(&'a mut self, idx : &[usize]) -> &'a mut T {
        if idx.len() != self.shape().len() {
            panic!("NDDataMut::idx_mut({:?}): idx is not of the right dimension ({} != {})", idx, idx.len(), self.dim());
        }
        let mut pos = 0usize;
        for i in 0..idx.len() {
            if idx[i] >= self.shape()[i] {
                panic!("NDDataMut::idx_mut({:?}): idx is out of bound for dimension {} (shape: {:?})", 
                        idx, i, self.shape());
            }
            pos += idx[i] * self.strides()[i];
        }
        return &mut self.get_data_mut()[pos];
    }

    fn transpose(&mut self) {
        // Check shape
        for i in 0..self.dim() {
            if self.shape()[0] != self.shape()[i] {
                panic!(format!("NDDataMut::transpose(): generic transpose only apply to data where all the dimensions length are the same, shape[0] != shape[{}] ({} != {})",
                                i, self.shape()[0], self.shape()[i]));
            }
        }

        let mut idx : Vec<usize>= repeat(0usize).take(self.dim()).collect();
        let copy = NDArray::<T>::from_slice(&self.shape()[..], self.get_data());
        loop {
            let revidx : Vec<usize> = idx.iter().rev().cloned().collect();
            *self.idx_mut(&idx[..]) = copy.idx(&revidx[..]).clone();
            // Update idx
            let mut i = 0;
            while i < idx.len() {
                idx[i] += 1;
                if idx[i] >= self.shape()[i] {
                        idx[i] = 0;
                    i += 1;
                }
                else {
                    break;
                }
            }
            if i == idx.len() {
                break;
            }
        }
    }

    fn assign(&mut self, other : &NDData<T>) {
        if self.dim() != self.dim() {
            panic!("NDDataMut::assign(): other is not of the same dimension ({} != {})",  other.dim(), self.dim());
        }

        if self.shape() != other.shape() {
            panic!("NDDataMut::assign(): other is not of the same shape ({:?} != {:?})",  other.shape(), self.shape());
        }

        let mut idx : Vec<usize>= repeat(0usize).take(self.dim()).collect();
        loop {
            *self.idx_mut(&idx[..]) = other.idx(&idx[..]).clone();
            // Update idx
            let mut i = 0;
            while i < idx.len() {
                idx[i] += 1;
                if idx[i] >= self.shape()[i] {
                        idx[i] = 0;
                    i += 1;
                }
                else {
                    break;
                }
            }
            if i == idx.len() {
                break;
            }
        }
    }
}

pub trait NDSliceable<'a, T : 'a> {

    fn slice(&'a self, idx : &[usize]) -> NDSlice<'a, T>;
}

pub trait NDSliceableMut<'a, T : 'a> : NDSliceable<'a, T> {

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

pub struct NDArray<T> {
    shape : Vec<usize>,
    strides : Vec<usize>,
    data : Box<[T]>,
}

impl<T : Clone> NDArray<T> {

    fn compute_strides(shape : &[usize]) -> Vec<usize> {
        let mut strides : Vec<usize> = repeat(0usize).take(shape.len()).collect();
        let mut size = 1usize;
        for i in 0..shape.len() {
            let revidx = shape.len() - i - 1;
            strides[revidx] = size;
            size *= shape[revidx];
        }
        return strides;
    }

    pub fn new(shape : &[usize], v : T) -> NDArray<T> {
        let size = shape.iter().fold(1usize, |acc, &x| acc * x);
        let alloc : Vec<T> = repeat(v).take(size).collect();

        return NDArray {
            shape : shape.to_vec(),
            strides : NDArray::<T>::compute_strides(&shape),
            data : alloc.into_boxed_slice(),
        }
    }

    pub fn copy<R : NDData<T>>(data : &R) -> NDArray<T> {
        NDArray {
            shape : data.shape().to_vec(),
            strides : data.strides().to_vec(),
            data : data.get_data().to_vec().into_boxed_slice(),
        }
    }

    pub fn from_slice(shape : &[usize], data : &[T]) -> NDArray<T> {
        let shape_size = shape.iter().fold(1usize, |acc, &x| acc * x);
        if shape_size != data.len() {
            panic!(format!("NDArray::from_slice({:?}, data) : The size of the shape ({}) and the data ({}) doesn't match", shape, shape_size, data.len()));
        }
        NDArray {
            shape : shape.to_vec(),
            strides : NDArray::<T>::compute_strides(&shape),
            data : data.to_vec().into_boxed_slice(),
        }
    }

    pub fn reshape(&mut self, new_shape : &[usize]) -> Result<(),String> {
        let size1 = self.shape.iter().fold(1usize, |acc, &x| acc * x);
        let size2 = new_shape.iter().fold(1usize, |acc, &x| acc * x);

        if size1 != size2 {
            return Err(format!("New shape has a different size than the previous shape: {} != {}", size1, size2));
        }
        self.shape = new_shape.to_vec();
        self.strides = NDArray::<T>::compute_strides(&new_shape);
        return Ok(());
    }
}


impl<T> NDData<T> for NDArray<T> { 

    fn shape(&self) -> &[usize] {
        &self.shape[..]
    }

    fn strides(&self) -> &[usize] {
        &self.strides[..]
    }

    fn get_data(&self) -> &[T] {
        &self.data[..]
    }
}

impl<T : Clone + Display> NDDataMut<T> for NDArray<T> { 

    fn get_data_mut(&mut self) -> &mut [T] {
        &mut self.data[..]
    }

    fn transpose(&mut self) {
        let copy = NDArray::<T>::copy(self);
        let mut idx : Vec<usize>= repeat(0usize).take(self.dim()).collect();
        self.shape.reverse();
        self.strides = NDArray::<T>::compute_strides(&self.shape[..]);
        loop {
            let revidx : Vec<usize> = idx.iter().rev().cloned().collect();
            *self.idx_mut(&idx[..]) = copy.idx(&revidx[..]).clone();
            // Update idx
            let mut i = 0;
            while i < idx.len() {
                idx[i] += 1;
                if idx[i] >= self.shape()[i] {
                        idx[i] = 0;
                    i += 1;
                }
                else {
                    break;
                }
            }
            if i == idx.len() {
                break;
            }
        }
    }
}

impl<'a, T : 'a> NDSliceable<'a, T> for NDArray<T> {

    fn slice(&'a self, idx : &[usize]) -> NDSlice<'a, T> {
        if idx.len() >= self.shape().len() {
            panic!("NDArray::slice({:?}): idx is not of the right dimension ({} >= {})", idx, idx.len(), self.dim());
        }
        let mut start = 0usize;
        for i in 0..idx.len() {
            if idx[i] >= self.shape()[i] {
                panic!("NDArray::slice({:?}): idx is out of bound for dimension {} (shape: {:?})", 
                        idx, i, self.shape());
            }
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

impl<'a, T : 'a> NDSliceableMut<'a, T> for NDArray<T> {

    fn slice_mut(&'a mut self, idx : &[usize]) -> NDSliceMut<'a, T> {
        if idx.len() >= self.shape().len() {
            panic!("NDArray::slice_mut({:?}): idx is not of the right dimension ({} >= {})", idx, idx.len(), self.dim());
        }
        let mut start = 0usize;
        for i in 0..idx.len() {
            if idx[i] >= self.shape()[i] {
                panic!("NDArray::slice_mut({:?}): idx is out of bound for dimension {} (shape: {:?})", 
                        idx, i, self.shape());
            }
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

impl<'b, T> Index<&'b [usize]> for NDArray<T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize]) -> &'c T {
        self.idx(idx)
    }
}

impl<'b, T> Index<&'b [usize;1]> for NDArray<T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize;1]) -> &'c T {
        self.idx(idx)
    }
}

impl<'b, T> Index<&'b [usize;2]> for NDArray<T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize;2]) -> &'c T {
        self.idx(idx)
    }
}

impl<'b, T> Index<&'b [usize;3]> for NDArray<T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize;3]) -> &'c T {
        self.idx(idx)
    }
}

impl<'b, T : Clone + Display> IndexMut<&'b [usize]> for NDArray<T> {

    fn index_mut<'c>(&'c mut self, idx : &[usize]) -> &'c mut T {
        self.idx_mut(idx)
    }
}

impl<'b, T : Clone + Display> IndexMut<&'b [usize;1]> for NDArray<T> {

    fn index_mut<'c>(&'c mut self, idx : &[usize;1]) -> &'c mut T {
        self.idx_mut(idx)
    }
}

impl<'b, T : Clone + Display> IndexMut<&'b [usize;2]> for NDArray<T> {

    fn index_mut<'c>(&'c mut self, idx : &[usize;2]) -> &'c mut T {
        self.idx_mut(idx)
    }
}

impl<'b, T : Clone + Display> IndexMut<&'b [usize;3]> for NDArray<T> {

    fn index_mut<'c>(&'c mut self, idx : &[usize;3]) -> &'c mut T {
        self.idx_mut(idx)
    }
}

impl<'a, T> NDData<T> for NDSlice<'a, T> {

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn get_data(&self) -> &[T] {
        self.data
    }
}

impl<'a, T : 'a> NDSliceable<'a, T> for NDSlice<'a, T> {

    fn slice(&'a self, idx : &[usize]) -> NDSlice<'a, T> {
        if idx.len() >= self.shape().len() {
            panic!("NDSlice::slice({:?}): idx is not of the right dimension ({} >= {})", idx, idx.len(), self.dim());
        }
        let mut start = 0usize;
        for i in 0..idx.len() {
            if idx[i] >= self.shape()[i] {
                panic!("NDSlice::slice({:?}): idx is out of bound for dimension {} (shape: {:?})", 
                        idx, i, self.shape());
            }
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
        self.idx(idx)
    }
}

impl<'a, 'b, T> Index<&'b [usize;1]> for NDSlice<'a, T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize;1]) -> &'c T {
        self.idx(idx)
    }
}

impl<'a, 'b, T> Index<&'b [usize;2]> for NDSlice<'a, T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize;2]) -> &'c T {
        self.idx(idx)
    }
}

impl<'a, 'b, T> Index<&'b [usize;3]> for NDSlice<'a, T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize;3]) -> &'c T {
        self.idx(idx)
    }
}

impl<'a, T> NDData<T> for NDSliceMut<'a, T> {

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn get_data(&self) -> &[T] {
        self.data
    }
}

impl<'a, T : Clone + Display> NDDataMut<T> for NDSliceMut<'a, T> {

    fn get_data_mut(&mut self) -> &mut [T] {
        self.data
    }
}

impl<'a, T : 'a> NDSliceable<'a, T> for NDSliceMut<'a, T> {

    fn slice(&'a self, idx : &[usize]) -> NDSlice<'a, T> {
        if idx.len() >= self.shape().len() {
            panic!("NDSliceMut::slice({:?}): idx is not of the right dimension ({} >= {})", idx, idx.len(), self.dim());
        }
        let mut start = 0usize;
        for i in 0..idx.len() {
            if idx[i] >= self.shape()[i] {
                panic!("NDSliceMut::slice({:?}): idx is out of bound for dimension {} (shape: {:?})", 
                        idx, i, self.shape());
            }
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
        if idx.len() >= self.shape().len() {
            panic!("NDSliceMut::slice_mut({:?}): idx is not of the right dimension ({} >= {})", idx, idx.len(), self.dim());
        }
        let mut start = 0usize;
        for i in 0..idx.len() {
            if idx[i] >= self.shape()[i] {
                panic!("NDSliceMut::slice_mut({:?}): idx is out of bound for dimension {} (shape: {:?})", 
                        idx, i, self.shape());
            }
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
        self.idx(idx)
    }
}

impl<'a, 'b, T> Index<&'b [usize;1]> for NDSliceMut<'a, T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize;1]) -> &'c T {
        self.idx(idx)
    }
}

impl<'a, 'b, T> Index<&'b [usize;2]> for NDSliceMut<'a, T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize;2]) -> &'c T {
        self.idx(idx)
    }
}

impl<'a, 'b, T> Index<&'b [usize;3]> for NDSliceMut<'a, T> {
    type Output = T;

    fn index<'c>(&'c self, idx : &'b [usize;3]) -> &'c T {
        self.idx(idx)
    }
}

impl<'a, 'b, T : Clone + Display> IndexMut<&'b [usize]> for NDSliceMut<'a, T> {

    fn index_mut<'c>(&'c mut self, idx : &[usize]) -> &'c mut T {
        self.idx_mut(idx)
    }
}

impl<'a, 'b, T : Clone + Display> IndexMut<&'b [usize;1]> for NDSliceMut<'a, T> {

    fn index_mut<'c>(&'c mut self, idx : &'b [usize;1]) -> &'c mut T {
        self.idx_mut(idx)
    }
}

impl<'a, 'b, T : Clone + Display> IndexMut<&'b [usize;2]> for NDSliceMut<'a, T> {

    fn index_mut<'c>(&'c mut self, idx : &'b [usize;2]) -> &'c mut T {
        self.idx_mut(idx)
    }
}

impl<'a, 'b, T : Clone + Display> IndexMut<&'b [usize;3]> for NDSliceMut<'a, T> {

    fn index_mut<'c>(&'c mut self, idx : &'b [usize;3]) -> &'c mut T {
        self.idx_mut(idx)
    }
}

impl<T : PartialEq> PartialEq for NDArray<T> {

    fn eq(&self, other: &NDArray<T>) -> bool {
        if self.dim() != other.dim() {
            return false;
        }

        for i in 0..self.dim() {
            if self.shape()[i] != other.shape()[i] {
                return false;
            }
        }

        let mut idx : Vec<usize>= repeat(0usize).take(self.dim()).collect();
        loop {
            if *self.idx(&idx[..]) != *other.idx(&idx[..]) {
                return false;
            }
            // Update idx
            let mut i = 0;
            while i < idx.len() {
                idx[i] += 1;
                if idx[i] >= self.shape()[i] {
                        idx[i] = 0;
                    i += 1;
                }
                else {
                    break;
                }
            }
            if i == idx.len() {
                break;
            }
        }
        return true;
    }
}

impl<T : Eq> Eq for NDArray<T> {
}

impl<'a, T : PartialEq> PartialEq for NDSlice<'a, T> {

    fn eq(&self, other: &NDSlice<'a, T>) -> bool {
        if self.dim() != other.dim() {
            return false;
        }

        for i in 0..self.dim() {
            if self.shape()[i] != other.shape()[i] {
                return false;
            }
        }

        let mut idx : Vec<usize>= repeat(0usize).take(self.dim()).collect();
        loop {
            if *self.idx(&idx[..]) != *other.idx(&idx[..]) {
                return false;
            }
            // Update idx
            let mut i = 0;
            while i < idx.len() {
                idx[i] += 1;
                if idx[i] >= self.shape()[i] {
                        idx[i] = 0;
                    i += 1;
                }
                else {
                    break;
                }
            }
            if i == idx.len() {
                break;
            }
        }
        return true;
    }
}

impl<'a, T : Eq> Eq for NDSlice<'a, T> {
}

impl<'a, T : PartialEq> PartialEq for NDSliceMut<'a, T> {

    fn eq(&self, other: &NDSliceMut<'a, T>) -> bool {
        if self.dim() != other.dim() {
            return false;
        }

        for i in 0..self.dim() {
            if self.shape()[i] != other.shape()[i] {
                return false;
            }
        }

        let mut idx : Vec<usize>= repeat(0usize).take(self.dim()).collect();
        loop {
            if *self.idx(&idx[..]) != *other.idx(&idx[..]) {
                return false;
            }
            // Update idx
            let mut i = 0;
            while i < idx.len() {
                idx[i] += 1;
                if idx[i] >= self.shape()[i] {
                        idx[i] = 0;
                    i += 1;
                }
                else {
                    break;
                }
            }
            if i == idx.len() {
                break;
            }
        }
        return true;
    }
}

impl<'a, T : Eq> Eq for NDSliceMut<'a, T> {
}
