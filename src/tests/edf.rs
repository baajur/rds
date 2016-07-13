use array::{NDData, NDArray};
use array::edf::EDFFile;

#[test]
fn edf_read() {
    let file = EDFFile::new("test_vector/edf/test_generator.edf");
    for i in 0..16 {
        let array : NDArray<f32> = file.read_signal(i).unwrap();
    }
}

#[test]
fn edfplus_read() {
    let file = EDFFile::new("test_vector/edf/test_generator_2.edf");
    for i in 0..12 {
        let array : NDArray<f32> = file.read_signal(i).unwrap();
    }
}
