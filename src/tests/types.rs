use types::cast::Cast;
use types::complex::{c32,c64,Complex};

const F32_TRESH : f32 = 0.000001;
const F64_TRESH : f64 = 0.000000000001;

#[test]
fn c32() {
    let a = c32::cartesian(1.0, 2.0);
    let b = c32::cartesian(2.0, 1.0);
    let c = c32::cartesian(3.0, -4.0);

    assert_eq!(a * b, c32::cartesian(0.0, 5.0));
    assert_eq!(a + c, c32::cartesian(4.0, -2.0));
    assert_eq!(a - c, c32::cartesian(-2.0, 6.0));
    assert_eq!(a.conj(), c32::cartesian(1.0, -2.0));
    assert_eq!(a.abs(), (5.0f32).sqrt());
    assert!((a / b - c32::cartesian(0.8, 0.6)).abs() < F32_TRESH);
    assert!((c32::polar(a.abs(), a.arg()) - a).abs() < F32_TRESH);
}

#[test]
fn c64() {
    let a = c64::cartesian(1.0, 2.0);
    let b = c64::cartesian(2.0, 1.0);
    let c = c64::cartesian(3.0, -4.0);

    assert_eq!(a * b, c64::cartesian(0.0, 5.0));
    assert_eq!(a + c, c64::cartesian(4.0, -2.0));
    assert_eq!(a - c, c64::cartesian(-2.0, 6.0));
    assert_eq!(a.conj(), c64::cartesian(1.0, -2.0));
    assert_eq!(a.abs(), (5.0f64).sqrt());
    assert!((a / b - c64::cartesian(0.8, 0.6)).abs() < F64_TRESH);
    assert!((c64::polar(a.abs(), a.arg()) - a).abs() < F64_TRESH);
}

#[test]
fn cast() {
    let a = c64::cartesian(1.0, 2.0);
    assert!(Cast::<u8>::cast(a)  == 1u8);
    assert!(Cast::<u16>::cast(a) == 1u16);
    assert!(Cast::<u32>::cast(a) == 1u32);
    assert!(Cast::<u64>::cast(a) == 1u64);
    assert!(Cast::<i8>::cast(a)  == 1i8);
    assert!(Cast::<i16>::cast(a) == 1i16);
    assert!(Cast::<i32>::cast(a) == 1i32);
    assert!(Cast::<i64>::cast(a) == 1i64);
    assert!(Cast::<f32>::cast(a) == 1f32);
    assert!(Cast::<f64>::cast(a) == 1f64);
    assert!(Cast::<c32>::cast(a) == c32::cartesian(1.0, 2.0));
    assert!(Cast::<c64>::cast(a) == c64::cartesian(1.0, 2.0));
}
