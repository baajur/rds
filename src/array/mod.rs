use std::cmp::{PartialEq, Eq};
use std::iter::repeat;
use std::fmt::Display;
use std::ops::{Index, IndexMut};

/// N-dimensional indexing functionality.
pub mod ndindex;
/// CSV file support for loading and saving 1D and 2D arrays.
pub mod csv;
/// Numpy file support for loading and saving N-dimensional arrays.
pub mod numpy;
/// EDF file support for loading one dimensional signals.
pub mod edf;

use array::ndindex::NDIndex;
use types::cast::Cast;

/// A trait for struture giving immutable access to a N-dimensional array of type T
pub trait NDData<T>{

    /// Return a slice of length N where each element is the length of the dimension.
    /// For a 2 dimensional matrix, the first dimension is the number of rows and the second 
    /// dimension is the number of columns.
    fn shape(&self) -> &[usize];

    /// Return a slice of length N where each element is the stride of the dimension.
    fn strides(&self) -> &[usize]; 

    /// Return the underlying storage array as a slice.
    fn get_data(&self) -> &[T];

    /// Return N, the number of dimensions.
    fn dim(&self) -> usize {
        self.shape().len()
    }

    /// Return the total number of element of the N-dimensional array.
    fn size(&self) -> usize {
        self.shape().iter().fold(1usize, |acc, &x| acc * x)
    }

    /// Take a slice of length N representing an N-dimensional index in the array and return a reference to 
    /// the element at this position.
    fn idx<'a>(&'a self, idx : &[usize]) -> &'a T {
        if idx.len() != self.shape().len() {
            panic!("NDData::idx({:?}): idx is not of the right dimension ({} != {})", 
                   idx, idx.len(), self.dim());
        }
        let pos = idx.to_pos(self.shape(), self.strides());
        return &self.get_data()[pos];
    }
}

/// A trait for struture giving mutable access to a N-dimensional array of type T.
pub trait NDDataMut<T : Clone + Display> : NDData<T> {

    /// Return the underlying storage array as a mutable slice.
    fn get_data_mut(&mut self) -> &mut [T];

    /// Take a slice of length N representing an -index in the array and return a mutable 
    /// reference to the element at this position.
    fn idx_mut<'a>(&'a mut self, idx : &[usize]) -> &'a mut T {
        if idx.len() != self.shape().len() {
            panic!("NDDataMut::idx_mut({:?}): idx is not of the right dimension ({} != {})", idx, idx.len(), self.dim());
        }
        let pos = idx.to_pos(self.shape(), self.strides());
        return &mut self.get_data_mut()[pos];
    }

    /// Perform a generic transpose. All the dimensions need to be the same.
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
            idx.inc_ro(self.shape());
            if idx.is_zero() {
                break;
            }
        }
    }

    /// Assign another NDData<T> to the NDDataMut<T>. The shapes need to be identical.
    fn assign(&mut self, other : &NDData<T>) {
        if self.dim() != self.dim() {
            panic!("NDDataMut::assign(): other is not of the same dimensionality ({} != {})",  other.dim(), self.dim());
        }

        if self.shape() != other.shape() {
            panic!("NDDataMut::assign(): other is not of the same shape ({:?} != {:?})",  other.shape(), self.shape());
        }

        let mut idx : Vec<usize>= repeat(0usize).take(self.dim()).collect();
        loop {
            *self.idx_mut(&idx[..]) = other.idx(&idx[..]).clone();
            // Update idx
            idx.inc_ro(self.shape());
            if idx.is_zero() {
                break;
            }
        }
    }
}

/// A trait for N-dimensional data which can be sliced into a immutable sub slice.
pub trait NDSliceable<'a, T : 'a> {

    /// Take a slice of length < N representing the sub slice index and return immutable borrow of 
    /// the sub slice as an NDSlice.
    /// Because the storage is in row-major order and the slice need to be contiguous in the 
    /// underlying storage array it means, for example, only the rows of a matrix can be borrowed.
    fn slice(&'a self, idx : &[usize]) -> NDSlice<'a, T>;
}

/// A trait for N-dimensional data which can be sliced into a mutable sub slice.
pub trait NDSliceableMut<'a, T : 'a> : NDSliceable<'a, T> {

    /// Take a slice of length < N representing the sub slice index and return immutable borrow of 
    /// the sub slice as an NDSlice.
    /// Because the storage is in row-major order and the slice need to be contiguous in the 
    /// underlying storage array it means, for example, only the rows of a matrix can be borrowed.
    fn slice_mut(&'a mut self, idx : &[usize]) -> NDSliceMut<'a, T>;
}

/// Structure representing an immutable borrow of a n-dimensional array sub slice.
pub struct NDSlice<'a, T : 'a> {
    shape : &'a [usize],
    strides : &'a [usize],
    data : &'a [T],
}

/// Structure representing an mutable borrow of a n-dimensional array sub slice.
pub struct NDSliceMut<'a, T : 'a> {
    shape : &'a [usize],
    strides : &'a [usize],
    data : &'a mut [T],
}

/// Structure representing an owned n-dimensional array. The underlying storage is in row-major 
/// order.
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

    /// Allocate a new array of the specified shape with all elements initialized with the value v.
    pub fn new(shape : &[usize], v : T) -> NDArray<T> {
        let size = shape.iter().fold(1usize, |acc, &x| acc * x);
        let alloc : Vec<T> = repeat(v).take(size).collect();

        return NDArray {
            shape : shape.to_vec(),
            strides : NDArray::<T>::compute_strides(&shape),
            data : alloc.into_boxed_slice(),
        }
    }

    /// Allocate a new array which is a copy of data.
    pub fn copy<R : NDData<T>>(data : &R) -> NDArray<T> {
        NDArray {
            shape : data.shape().to_vec(),
            strides : data.strides().to_vec(),
            data : data.get_data().to_vec().into_boxed_slice(),
        }
    }

    /// Allocate a new array from a row-major order contiguous array and a shape. The size of the 
    /// shape (product of all its elements) must be equal to the array length.
    pub fn from_slice(shape : &[usize], data : &[T]) -> NDArray<T> {
        let shape_size = shape.iter().fold(1usize, |acc, &x| acc * x);
        if shape_size != data.len() {
            panic!(format!("NDArray::from_slice({:?}, data): The size of the shape ({}) and the data ({}) doesn't match", shape, shape_size, data.len()));
        }
        NDArray {
            shape : shape.to_vec(),
            strides : NDArray::<T>::compute_strides(&shape),
            data : data.to_vec().into_boxed_slice(),
        }
    }
    
    /// Allocate a new array where each element has been casted from data.
    pub fn cast<U : Copy>(data : &NDData<U>) -> NDArray<T> where T : Copy, U : Cast<T> {
        let alloc : Vec<T> = data.get_data().iter().map(|x| Cast::<T>::cast(x.clone())).collect();
        NDArray {
            shape : data.shape().to_vec(),
            strides : data.strides().to_vec(),
            data : alloc.into_boxed_slice()
        }
    }

    /// Reshape this NDArray. The size of the new shape (product of all its elements) must be equal 
    /// to the size of the current shape.
    pub fn reshape(&mut self, new_shape : &[usize]) {
        let size1 = self.shape.iter().fold(1usize, |acc, &x| acc * x);
        let size2 = new_shape.iter().fold(1usize, |acc, &x| acc * x);

        if size1 != size2 {
            panic!(format!("NDArray::reshape({:?}): New shape has a different size than the previous shape: {} != {}", new_shape, size1, size2));
        }
        self.shape = new_shape.to_vec();
        self.strides = NDArray::<T>::compute_strides(&new_shape);
    }

    /// Insert another NDData of the same dimensionality in this NDArray. The insertion is made 
    /// at the position pos in the dimension dim. The shape of NDData need to be the same as 
    /// this NDArray in every dimension except for the insertion dimension.
    pub fn insert(&mut self, dim : usize, pos : usize, other : &NDData<T>) {
        if dim >= self.dim() {
            panic!("NDArray::insert(): dim is greater than array dimension ({} >= {})", dim, self.dim());
        }
        if pos > self.shape[dim] {
            panic!("NDArray::insert(): pos is out of bound ({} > {})", pos, self.shape[dim]);
        }
        if self.dim() != other.dim() {
            panic!("NDArray::insert(): dimensions are differents ({} != {})", other.dim(), self.dim());
        }
        for i in 0..self.dim() {
            if i != dim && self.shape[i] != other.shape()[i] {
                panic!("NDArray::insert(): Shapes are different at dimension {} ({} != {})", i, other.shape()[i], self.shape()[i]);
            }
        }

        let old_shape = self.shape.clone();
        let old_strides = self.strides.clone();
        let old_data = self.data.clone();
        self.shape[dim] += other.shape()[dim];
        self.strides = NDArray::<T>::compute_strides(&self.shape);
        let alloc : Vec<T> = repeat(old_data[0].clone()).take(self.size()).collect();
        self.data = alloc.into_boxed_slice();
        let mut index : Vec<usize> = repeat(0usize).take(self.dim()).collect();

        loop {
            if index[dim] < pos {
                self.data[index.to_pos(&self.shape, &self.strides)] = old_data[index.to_pos(&old_shape, &old_strides)].clone();
            }
            else if index[dim] >= pos && index[dim] < pos + other.shape()[dim] {
                index[dim] -= pos;
                let v = other.idx(&index[..]).clone();
                index[dim] += pos;
                self.data[index.to_pos(&self.shape, &self.strides)] = v;
            }
            else {
                index[dim] -= other.shape()[dim];
                let v = old_data[index.to_pos(&old_shape, &old_strides)].clone();
                index[dim] += other.shape()[dim];
                self.data[index.to_pos(&self.shape, &self.strides)] = v;
            }

            index.inc_ro(&self.shape);
            if index.is_zero() {
                break;
            }
        }
    }

    /// Extract a sub part of this NDArray as a new NDArray. start and end are two index of the 
    /// same dimensionality ad this NDArray.
    pub fn extract(&self, start : &[usize], end : &[usize]) -> NDArray<T> {
        if self.dim() != start.len() || self.dim() != end.len() {
            panic!("NDArray::extract(): start or end dimension is not the same as array dimension ({} != {} || {} != {})", start.len(), self.dim(), end.len(), self.dim());
        }
        for i in 0..self.dim() {
            if start[i] >= self.shape[i] {
                panic!("NDArray::extract(): start is out of bound at dimension {} ({} >= {})", i, start[i], self.shape[i]);
            }
            if end[i] > self.shape[i] {
                panic!("NDArray::extract(): end is out of bound at dimension {} ({} > {})", i, end[i], self.shape[i]);
            }
            if start[i] >= end[i] {
                panic!("NDArray::extract(): start is not before end at dimension {} ({} >= {})", i, start[i], end[i]);
            }
        }

        let mut shape = end.to_vec();
        for i in 0..shape.len() {
            shape[i] -= start[i];
        }
        let strides = NDArray::<T>::compute_strides(&shape);
        let mut data : Vec<T> = repeat(self.data[0].clone()).take(shape.iter().fold(1usize, |acc, &x| acc * x)).collect();
        let mut index : Vec<usize> = repeat(0usize).take(self.dim()).collect();
        let mut self_index = index.clone();

        loop {
            for i in 0..shape.len() {
                self_index[i] = index[i] + start[i];
            }
            data[index.to_pos(&shape, &strides)] = self.idx(&self_index).clone();
            index.inc_ro(&shape);
            if index.is_zero() {
                break;
            }
        }

        return NDArray::<T> {
            shape : shape,
            strides : strides,
            data : data.into_boxed_slice()
        };
    }

    /// Remove a part of this NDArray. The removal is made in dimension dim from the position start 
    /// until end (not included). This can be seen as the opposite operation of insert.
    pub fn remove(&mut self, dim : usize, start : usize, end : usize) {
        if dim >= self.dim() {
            panic!("NDArray::remove(): dim is greater than array dimension ({} >= {})", dim, self.dim());
        }
        if start >= self.shape[dim] || end > self.shape[dim]{
            panic!("NDArray::remove(): start or end is out of bound ({} >= {} || {} > {})", start, self.shape[dim], end, self.shape[dim]);
        }
        if start >= end {
            panic!("NDArray::remove(): start is not before end ({} >= {})", start, end);
        }

        let old_shape = self.shape.clone();
        let old_strides = self.strides.clone();
        let old_data = self.data.clone();
        self.shape[dim] -= end - start;
        self.strides = NDArray::<T>::compute_strides(&self.shape);
        let alloc : Vec<T> = repeat(old_data[0].clone()).take(self.size()).collect();
        self.data = alloc.into_boxed_slice();
        let mut index : Vec<usize> = repeat(0usize).take(self.dim()).collect();

        loop {
            if index[dim] < start {
                self.data[index.to_pos(&self.shape, &self.strides)] = old_data[index.to_pos(&old_shape, &old_strides)].clone();
            }
            else {
                index[dim] += end - start;
                let v = old_data[index.to_pos(&old_shape, &old_strides)].clone();
                index[dim] -= end - start;
                self.data[index.to_pos(&self.shape, &self.strides)] = v;
            }

            index.inc_ro(&self.shape);
            if index.is_zero() {
                break;
            }
        }
    }

    /// Split this NDArray at position pos of dimension dim. The lower indices are kept in dim 
    /// while the upper indices are returned. This is in fact an extract operation followed by a 
    /// remove operation.
    pub fn split(&mut self, dim : usize, pos : usize) -> NDArray<T> {
        if dim >= self.dim() {
            panic!("NDArray::split(): dim is greater than array dimension ({} >= {})", dim, self.dim());
        }
        let end = self.shape.clone();
        let mut start : Vec<usize> = repeat(0usize).take(self.dim()).collect();
        start[dim] = pos;
        let splitted = self.extract(&start[..], &end[..]);
        self.remove(dim, start[dim], end[dim]);
        return splitted;
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

    /// The transpose function of a NDArray allow to transpose any shape.
    fn transpose(&mut self) {
        let copy = NDArray::<T>::copy(self);
        let mut idx : Vec<usize>= repeat(0usize).take(self.dim()).collect();
        self.shape.reverse();
        self.strides = NDArray::<T>::compute_strides(&self.shape[..]);
        loop {
            let revidx : Vec<usize> = idx.iter().rev().cloned().collect();
            *self.idx_mut(&idx[..]) = copy.idx(&revidx[..]).clone();
            // Update idx
            idx.inc_ro(self.shape());
            if idx.is_zero() {
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

impl<T : PartialEq, O : NDData<T> + Sized> PartialEq<O> for NDArray<T> {

    fn eq(&self, other: &O) -> bool {
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
            idx.inc_ro(self.shape());
            if idx.is_zero() {
                break;
            }
        }
        return true;
    }
}

impl<T : Eq> Eq for NDArray<T> {
}

impl<'a, T : PartialEq, O : NDData<T> + Sized> PartialEq<O> for NDSlice<'a, T> {

    fn eq(&self, other: &O) -> bool {
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
            idx.inc_ro(self.shape());
            if idx.is_zero() {
                break;
            }
        }
        return true;
    }
}

impl<'a, T : Eq> Eq for NDSlice<'a, T> {
}

impl<'a, T : PartialEq, O : NDData<T> + Sized> PartialEq<O> for NDSliceMut<'a, T> {

    fn eq(&self, other: &O) -> bool {
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
            idx.inc_ro(self.shape());
            if idx.is_zero() {
                break;
            }
        }
        return true;
    }
}

impl<'a, T : Eq> Eq for NDSliceMut<'a, T> {
}
