use std::iter::repeat;
use std::sync::{Arc, Mutex};

pub struct Array<T> {
    shape : Vec<usize>,
    strides : Vec<usize>,
    size : usize,
    data : Arc<Mutex<Box<[T]>>>,
}

impl<T : Clone> Array<T> {
    pub fn new(shape : Vec<usize>, v : T) -> Array<T> {
        let mut strides : Vec<usize> = repeat(0usize).take(shape.len()-1).collect();
        let mut size = 1usize;
        for i in 0..shape.len() {
            let revidx = shape.len() - i;
            size *= shape[revidx];
            if revidx > 0 {
                strides[revidx - 1] = size;
            }
        }
        let alloc : Vec<T> = repeat(v).take(size).collect();

        return Array {
            shape : shape.clone(),
            strides : strides,
            size : size,
            data : Arc::new(Mutex::new(alloc.into_boxed_slice())),
        }
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn length(&self, n : usize) -> usize {
        self.shape[n]
    }
}
