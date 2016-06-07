use array::{Array, NDSliceable, NDSliceableMut};

#[test]
fn indexing() {
    let mut array = Array::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                array[&[i, j, k][..]] = (i * j * k) as f64;
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                assert!(array[&[i,j,k][..]] == (i * j * k) as f64);
            }
        }
    }
}

#[test]
#[should_panic]
fn indexing_wrongdim() {
    let mut array = Array::<f64>::new(&[3, 3, 3], 0.0);
    array[&[1,2][..]] = 1.0;
}

#[test]
#[should_panic]
fn indexing_outofbound() {
    let mut array = Array::<f64>::new(&[3, 3, 3], 0.0);
    array[&[1,4,1][..]] = 1.0;
}

#[test]
fn slicing() {
    let mut array = Array::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                array[&[i, j, k][..]] = (i * j * k) as f64;
            }
        }
    }
    for i in 0..3 {
        let slice2d = array.slice(&[i]);
        for j in 0..3 {
            let slice1d = slice2d.slice(&[j]);
            for k in 0..3 {
                assert!(slice1d[&[k][..]] == (i * j * k) as f64);
            }
        }
    }
}

#[test]
fn copy() {
    let mut array = Array::<f64>::new(&[3, 3, 3], 0.0);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                array[&[i, j, k][..]] = (i * j * k) as f64;
            }
        }
    }
    let mut array2 = Array::copy(array.slice(&[2]));
    for i in 0..3 {
        for j in 0..3 {
            assert!(array2[&[i,j][..]] == (i * j * 2) as f64);
        }
    }
}
