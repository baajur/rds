array
=====

* Reshape (e.g 1d to 2d column vector or row vector)
* Transpose
* Join, extract, remove, split (extract + remove)
* Add idx function to NDData
* Add NDDataMut trait
* Add idx_mut function to NDDataMut
* Implement index operation for &[usize;1], &[usize;2], &[usize;3] as a temporary fix
* Make NDSliceableMut require NDSliceable
* Make NDDataMut require NDData
* Equality operator for NDData
* Assignation operator for NDDataMut

array::csv
==========

* Array iterator for CSV file
* read_row_iterator and write_row_iterator

array::numpy
============

* Reverse format
