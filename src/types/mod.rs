pub mod complex;
pub mod cast;

use types::complex::{c32, c64};
use types::cast::Cast;

pub enum RDSType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    C32,
    C64
}

pub trait RDSTyped : Cast<u8> + Cast<u16> + Cast<u32> + Cast<u64> + Cast<i8> + Cast<i16> + Cast<i32> + Cast<i64> + Cast<f32> + Cast<f64> + Cast<c32> + Cast<c64> {
    fn t() -> RDSType;
}

impl RDSTyped for u8 {
    fn t() -> RDSType {
        RDSType::U8
    }
}

impl RDSTyped for u16 {
    fn t() -> RDSType {
        RDSType::U16
    }
}

impl RDSTyped for u32 {
    fn t() -> RDSType {
        RDSType::U32
    }
}

impl RDSTyped for u64 {
    fn t() -> RDSType {
        RDSType::U64
    }
}

impl RDSTyped for i8 {
    fn t() -> RDSType {
        RDSType::I8
    }
}

impl RDSTyped for i16 {
    fn t() -> RDSType {
        RDSType::I16
    }
}

impl RDSTyped for i32 {
    fn t() -> RDSType {
        RDSType::I32
    }
}

impl RDSTyped for i64 {
    fn t() -> RDSType {
        RDSType::I64
    }
}

impl RDSTyped for f32 {
    fn t() -> RDSType {
        RDSType::F32
    }
}

impl RDSTyped for f64 {
    fn t() -> RDSType {
        RDSType::F64
    }
}

impl RDSTyped for c32 {
    fn t() -> RDSType {
        RDSType::C32
    }
}

impl RDSTyped for c64 {
    fn t() -> RDSType {
        RDSType::C64
    }
}
