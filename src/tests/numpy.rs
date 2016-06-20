use std::fs::read_dir;
use std::process::Command;

use array::{NDData, NDArray};
use array::numpy::NumpyFile;

#[test]
fn read() {
    const TMP_DIR : &'static str = "/tmp/rds_numpy_1/";
    assert!(Command::new("python").arg("test_vector/numpy/generate.py").arg(TMP_DIR).status().unwrap().success());
    for file in read_dir(TMP_DIR).unwrap() {
        let path = file.unwrap().path();
        let path_str = path.to_str().unwrap();
        if path_str.ends_with(".npy") {
            print!("Reading {}: ", path_str);
            let mut numpyfile = NumpyFile::new(path_str);
            let array : NDArray<f32> = numpyfile.read_array().unwrap();
            assert!(array.dim() <= 3);
            if array.dim() == 1 {
                for i in 0..array.shape()[0] {
                    assert!(array[&[i]] == (i*3) as f32);
                }
            }
            if array.dim() == 2 {
                for i in 0..array.shape()[0] {
                    for j in 0..array.shape()[1] {
                        assert!(array[&[i,j]] == (i*3 + j*5) as f32);
                    }
                }
            }
            if array.dim() == 3 {
                for i in 0..array.shape()[0] {
                    for j in 0..array.shape()[1] {
                        for k in 0..array.shape()[2] {
                            assert!(array[&[i,j,k]] == (i*3 + j*5 + k*7) as f32);
                        }
                    }
                }
            }
            println!("Ok");
        }
    }
}

#[test]
fn write() {
    const TMP_DIR : &'static str = "/tmp/rds_numpy_2/";
    assert!(Command::new("python").arg("test_vector/numpy/generate.py").arg(TMP_DIR).status().unwrap().success());
    for file in read_dir(TMP_DIR).unwrap() {
        let path = file.unwrap().path();
        let path_str = path.to_str().unwrap();
        if path_str.ends_with(".npy") {
            let mut numpyfile = NumpyFile::new(path_str);
            let array : NDArray<f32> = numpyfile.read_array().unwrap();
            assert!(array.dim() <= 3);
            if array.dim() == 1 {
                for i in 0..array.shape()[0] {
                    assert!(array[&[i]] == (i*3) as f32);
                }
            }
            if array.dim() == 2 {
                for i in 0..array.shape()[0] {
                    for j in 0..array.shape()[1] {
                        assert!(array[&[i,j]] == (i*3 + j*5) as f32);
                    }
                }
            }
            if array.dim() == 3 {
                for i in 0..array.shape()[0] {
                    for j in 0..array.shape()[1] {
                        for k in 0..array.shape()[2] {
                            assert!(array[&[i,j,k]] == (i*3 + j*5 + k*7) as f32);
                        }
                    }
                }
            }
            numpyfile.write_data(&array).unwrap();
        }
    }
    assert!(Command::new("python").arg("test_vector/numpy/verify.py").arg(TMP_DIR).status().unwrap().success());
}
