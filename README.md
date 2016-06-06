RDS
===

Rust Data Science is an attempt at bringing a solid basis for doing data science under rust. 
There already exist several scientific / data science library for rust but I'm not satisfied with 
their design. Following the [famous xkcd joke](https://xkcd.com/927/), this is just another attempt.

Current attempt lack I/O capacity (array file format reading/writting and plotting mostly) and thus 
require additional glue to be used in real world projects. They often try to reimplement everything 
in rust (even basic BLAS operations) and thus lack performances (efficient matrix multiplication is 
*hard*). Nowadays every data science library should offer transparent multithreading and gpu 
support but existing implementation either have architecture problem because it was not planned 
early enough or is limited to one specific vendor solution. 

Below is my plan, I know it's ambitious and I have no ETA.

Plan
----

* array  : n-dimensional arrays with common file format (csv, numpy, hdf5, matlab matrices) support.
* blas   : rusty blas abstraction for arrays which allow to transparently use BLAS, clBLAS, cuBLAS.
* fft    : fft abstraction for arrays which allow to transparently use fftw, clFFT, cuFFT.
* plot   : basic plotting library to output jpeg, png, svg plots.

On this basis we can hope to develop additional packages for linear algebra, statistics, 
signal processing, machine learning, neural networks.

Does it work ?
--------------

No, try [rusty-machine](https://github.com/AtheMathmo/rusty-machine) or 
[rust-ndarray](https://github.com/bluss/rust-ndarray).
