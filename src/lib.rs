/// Module containing N-dimensional array storage, manipulation and file format support.
/// 
/// # Introduction
/// 
/// N-dimensional arrays are stored in `NDArray` structs. Slice of those arrays can be borrowed 
/// using the `NDSliceable` and `NDSliceableMut` trait which returns `NDSlice` and `NDSliceMut` 
/// respectively. Those slice can also be sliced into sub slice.
///
/// Those N-dimensional structures implement two basic traits, `NDData` and `NDDataMut`, which give 
/// access to their data.
/// 
/// The `NDIndex` trait provides helper functions to manipulate N-dimensional indexes.
///
/// The CSV and Numpy sub module allow to load and store N-dimensional arrays.
///
/// # Examples
/// 
/// ## NDData
///
/// `NDArray`, `NDSlice` and `NDSliceMut` all implement `NDData`:
///
/// ```
/// use rds::array::{NDData, NDSliceable, NDArray, NDSlice};
///
/// let array = NDArray::<f32>::new(&[4, 3], 1.0);
///
/// assert!(array.dim() == 2);
/// assert!(array.shape() == &[4, 3]);
/// assert!(array.strides() == &[3, 1]);
/// assert!(array.size() == 12);
///
/// for i in 0..array.shape()[0] {
///     for j in 0..array.shape()[1] {
///         // Indexing can use the idx method or the [] operator
///         assert!(*array.idx(&[i,j]) == 1.0);
///         assert!(array[&[i,j]] == 1.0);
///     }
/// }   
///
/// // We borrow the second row as a NDSlice which also implements NDData
/// let row = array.slice(&[1]); 
/// 
/// assert!(row.dim() == 1);
/// assert!(row.shape() == &[3]);
/// assert!(row.strides() == &[1]);
/// assert!(row.size() == 3);
///
/// for i in 0..row.shape()[0] {
///     assert!(*row.idx(&[i]) == 1.0);
///     assert!(row[&[i]] == 1.0);
/// }
/// 
/// // NDData also overload the equality operator
/// assert!(row == NDArray::<f32>::new(&[3], 1.0));
/// assert!(row != NDArray::<f32>::new(&[3], 0.0));
/// assert!(row != NDArray::<f32>::new(&[4], 1.0));
/// ```
/// ## NDDataMut
/// 
/// `NDArray` and `NDSliceMut` implement NDDataMut:
///
/// ```
/// use rds::array::{NDData, NDDataMut, NDSliceableMut, NDArray, NDSliceMut};
///
/// let mut array = NDArray::<f32>::new(&[3, 3], 1.0);
///  
/// for i in 0..array.shape()[0] {
///     for j in 0..array.shape()[1] {
///         // Assignation using idx_mut 
///         *array.idx_mut(&[i,j]) = (i * 10 + j) as f32;
///     }
/// }
///
/// // mutable slicing and assign allow to set the first row to 0
/// array.slice_mut(&[0]).assign(&NDArray::<f32>::new(&[3], 0.0));
/// 
/// {
///     // Mutable borrow of the second row of array
///     let mut row = array.slice_mut(&[1]);
///     for i in 0..row.shape()[0] {
///         // Assignation using the [] operator
///         row[&[i]] = (row.shape()[0] - i) as f32;
///     }
/// }
/// 
/// // Transposition of the array, rows become columns
/// array.transpose();
/// 
/// // If you've followed everything until here
/// for i in 0..array.shape()[0] {
///     assert!(array[&[i, 0]] == 0.0);
///     assert!(array[&[i, 1]] == (array.shape()[0] - i) as f32);
///     assert!(array[&[i, 2]] == (20 + i) as f32);
/// }
/// ```
///
/// ## CSV
///
/// Here is how to load an array from a csv file, modify it then save it.
///
/// ```no_run
/// use rds::array::{NDData, NDDataMut, NDArray};
/// use rds::array::csv::CSVFile;
/// 
/// let mut csv_file = CSVFile::new("data.csv");
/// let mut array : NDArray<f32> = csv_file.read_array().unwrap();
/// for i in 0..array.shape()[0] {
///     for j in 0..array.shape()[1] {
///         array[&[i,j]] += 1.0;
///     }
/// }
/// csv_file.write_data(&array);
/// ```
///
/// ## Numpy
///
/// We can do the same with a numpy array for any number of dimensions using the `NDIndex` trait.
///
/// ```no_run
/// use std::iter::repeat;
/// use rds::array::{NDData, NDDataMut, NDArray};
/// use rds::array::ndindex::NDIndex;
/// use rds::array::numpy::NumpyFile;
/// 
/// let mut numpy_file = NumpyFile::new("data.npy");
/// let mut array : NDArray<f32> = numpy_file.read_array().unwrap();
/// // Allocate an index with the right number of dimensions
/// let mut idx : Vec<usize> = repeat(0usize).take(array.dim()).collect();
/// loop {
///     array[&idx[..]] += 1.0;
///     // Increment the index in row-major order
///     idx.inc_ro(array.shape());
///     // If the index overflow to zero, we looped through every elements
///     if idx.is_zero() {
///         break;
///     }
/// }
/// numpy_file.write_data(&array);
/// ```
pub mod array;

/// Module containing Blas bindings and overloaded operation for `NDData`.
///
/// # Introduction
/// 
/// Blas functionalities are implemented for any struct implementing `NDDataMut<T>` where `T` is 
/// either f32 or f64.
/// 
/// The convention is that the calling array is always the output array. This means for example 
/// that a matrix vector multiplication should be called on the vector. Whether a matrix need a 
/// translation or not is infered by the framework. If there are more than one solution, the one 
/// requiring the less translations is taken.
///
/// The operators are overloaded to use those Blas functions, however they only operate on 
/// references making the syntax a bit weird.
///
/// # Examples
///
/// ## Blas level 1
/// 
/// ```
/// use rds::array::{NDDataMut, NDArray};
/// use rds::blas::Blas;
/// 
/// let mut array1 = NDArray::<f32>::new(&[5], 1.0);
/// assert!(array1.asum() == 5.0);
/// assert!(array1.asum() == 5.0);
/// array1.scal(2.0);
/// let mut array2 = NDArray::<f32>::copy(&array1);
/// assert!(array1.dot(&array2) == 20.0);
/// array2 += &array1;
/// assert!(array2.asum() == 20.0);
/// ```
///
/// ## Blas level 2
/// 
/// ```
/// use rds::array::{NDDataMut, NDArray};
/// use rds::blas::Blas;
/// 
/// let vec1 = NDArray::<f32>::from_slice(&[2], &[2.0, 0.0]);
/// let mut vec2 = NDArray::<f32>::copy(&vec1);
/// let rot90 = NDArray::<f32>::from_slice(&[2,2], &[0.0, -1.0, 1.0, 0.0]);
/// vec2 *= &rot90;
/// assert!(vec1.dot(&vec2) == 0.0);
/// assert!(vec2 == NDArray::<f32>::from_slice(&[2], &[0.0, 2.0]));
/// ```
/// ## Blas level 3
///
///
/// ```
/// use rds::array::{NDDataMut, NDArray};
/// use rds::blas::Blas;
/// 
/// let rot90 = NDArray::<f32>::from_slice(&[2,2], &[0.0, -1.0, 1.0, 0.0]);
/// let rot180 = &rot90 * &rot90;
/// let identity = NDArray::<f32>::from_slice(&[2,2], &[1.0, 0.0, 0.0, 1.0]);
/// let mut transformed = NDArray::<f32>::copy(&identity);
/// transformed *= &rot90;
/// transformed *= &rot180;
/// transformed *= &rot90;
/// assert!(transformed == identity);
/// ```
pub mod blas;

#[cfg(test)]
mod tests;
