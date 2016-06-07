pub mod array;

#[cfg(test)]
mod tests {
    use array::{Array, NDSliceable, NDSliceableMut};
    use std::ops::IndexMut;
    #[test]
    fn it_works() {
        let mut array = Array::<f64>::new(vec![3, 3, 3], 0.0);
        array[&[1, 1, 1][..]] = 1.0;
        let slice = array.slice(&[1, 1]);
        assert!(slice[&[1][..]] == 1.0);
    }
}
