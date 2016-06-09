use array::{NDArray, NDData};
use array::csv::CSVFile;

#[test]
fn array() {
    let mut array = NDArray::<f64>::new(&[10, 5], 0.0);
    let mut csvfile = CSVFile::new("/tmp/test.csv");
    for i in 0..array.shape()[0] {
        for j in 0..array.shape()[1] {
            array[&[i, j]] = (i * 2 + j * 5) as f64;
        }
    }
    assert!(csvfile.write_data(&array) == Ok(()));
    let array2 : NDArray<f64> = csvfile.read_2darray().unwrap();
    assert!(array2.shape() == array.shape());
    assert!(array2.strides() == array.strides());
    for i in 0..array2.shape()[0] {
        for j in 0..array2.shape()[1] {
            assert!(array[&[i,j]] == array2[&[i,j]]);
        }
    }
}

#[test]
fn rows() {
    let mut array = NDArray::<f64>::new(&[10, 5], 0.0);
    let mut csvfile = CSVFile::new("/tmp/test.csv");
    for i in 0..array.shape()[0] {
        for j in 0..array.shape()[1] {
            array[&[i, j]] = (i * 2 + j * 5) as f64;
        }
    }
    assert!(csvfile.write_data(&array) == Ok(()));
    for i in 0..array.shape()[0] {
        let row : NDArray<f64> = csvfile.read_row(i).unwrap();
        for j in 0..row.shape()[0] {
            assert!(array[&[i,j]] == row[&[j]]);
        }
    }
}

#[test]
fn columns() {
    let mut array = NDArray::<f64>::new(&[10, 5], 0.0);
    let mut csvfile = CSVFile::new("/tmp/test.csv");
    for i in 0..array.shape()[0] {
        for j in 0..array.shape()[1] {
            array[&[i, j]] = (i * 2 + j * 5) as f64;
        }
    }
    assert!(csvfile.write_data(&array) == Ok(()));
    for j in 0..array.shape()[1] {
        let column : NDArray<f64> = csvfile.read_column(j).unwrap();
        for i in 0..column.shape()[0] {
            assert!(array[&[i,j]] == column[&[i]]);
        }
    }
}
