use types::complex::{c32,c64};

/// Trait allowing to cast Self into T.
pub trait Cast<T : Copy> {
    fn cast(self) -> T;
}

impl Cast<u8>  for u8  { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for u8  { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for u8  { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for u8  { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for u8  { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for u8  { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for u8  { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for u8  { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for u8  { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for u8  { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for u8  { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for u8  { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for u16 { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for u16 { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for u16 { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for u16 { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for u16 { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for u16 { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for u16 { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for u16 { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for u16 { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for u16 { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for u16 { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for u16 { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for u32 { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for u32 { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for u32 { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for u32 { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for u32 { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for u32 { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for u32 { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for u32 { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for u32 { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for u32 { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for u32 { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for u32 { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for u64 { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for u64 { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for u64 { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for u64 { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for u64 { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for u64 { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for u64 { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for u64 { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for u64 { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for u64 { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for u64 { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for u64 { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for i8  { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for i8  { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for i8  { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for i8  { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for i8  { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for i8  { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for i8  { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for i8  { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for i8  { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for i8  { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for i8  { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for i8  { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for i16 { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for i16 { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for i16 { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for i16 { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for i16 { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for i16 { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for i16 { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for i16 { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for i16 { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for i16 { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for i16 { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for i16 { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for i32 { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for i32 { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for i32 { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for i32 { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for i32 { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for i32 { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for i32 { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for i32 { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for i32 { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for i32 { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for i32 { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for i32 { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for i64 { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for i64 { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for i64 { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for i64 { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for i64 { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for i64 { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for i64 { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for i64 { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for i64 { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for i64 { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for i64 { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for i64 { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for f32 { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for f32 { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for f32 { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for f32 { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for f32 { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for f32 { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for f32 { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for f32 { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for f32 { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for f32 { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for f32 { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for f32 { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for f64 { fn cast(self) -> u8  { self as u8   } }
impl Cast<u16> for f64 { fn cast(self) -> u16 { self as u16  } }
impl Cast<u32> for f64 { fn cast(self) -> u32 { self as u32  } }
impl Cast<u64> for f64 { fn cast(self) -> u64 { self as u64  } }
impl Cast<i8>  for f64 { fn cast(self) -> i8  { self as i8   } }
impl Cast<i16> for f64 { fn cast(self) -> i16 { self as i16  } }
impl Cast<i32> for f64 { fn cast(self) -> i32 { self as i32  } }
impl Cast<i64> for f64 { fn cast(self) -> i64 { self as i64  } }
impl Cast<f32> for f64 { fn cast(self) -> f32 { self as f32  } }
impl Cast<f64> for f64 { fn cast(self) -> f64 { self as f64  } }
impl Cast<c32> for f64 { fn cast(self) -> c32 { c32::new(self as f32, 0.0) } }
impl Cast<c64> for f64 { fn cast(self) -> c64 { c64::new(self as f64, 0.0) } }

impl Cast<u8>  for c32 { fn cast(self) -> u8  { self.re as u8   } }
impl Cast<u16> for c32 { fn cast(self) -> u16 { self.re as u16  } }
impl Cast<u32> for c32 { fn cast(self) -> u32 { self.re as u32  } }
impl Cast<u64> for c32 { fn cast(self) -> u64 { self.re as u64  } }
impl Cast<i8>  for c32 { fn cast(self) -> i8  { self.re as i8   } }
impl Cast<i16> for c32 { fn cast(self) -> i16 { self.re as i16  } }
impl Cast<i32> for c32 { fn cast(self) -> i32 { self.re as i32  } }
impl Cast<i64> for c32 { fn cast(self) -> i64 { self.re as i64  } }
impl Cast<f32> for c32 { fn cast(self) -> f32 { self.re as f32  } }
impl Cast<f64> for c32 { fn cast(self) -> f64 { self.re as f64  } }
impl Cast<c32> for c32 { fn cast(self) -> c32 { self } }
impl Cast<c64> for c32 { fn cast(self) -> c64 { c64::new(self.re as f64, self.im as f64) } }

impl Cast<u8>  for c64 { fn cast(self) -> u8  { self.re as u8   } }
impl Cast<u16> for c64 { fn cast(self) -> u16 { self.re as u16  } }
impl Cast<u32> for c64 { fn cast(self) -> u32 { self.re as u32  } }
impl Cast<u64> for c64 { fn cast(self) -> u64 { self.re as u64  } }
impl Cast<i8>  for c64 { fn cast(self) -> i8  { self.re as i8   } }
impl Cast<i16> for c64 { fn cast(self) -> i16 { self.re as i16  } }
impl Cast<i32> for c64 { fn cast(self) -> i32 { self.re as i32  } }
impl Cast<i64> for c64 { fn cast(self) -> i64 { self.re as i64  } }
impl Cast<f32> for c64 { fn cast(self) -> f32 { self.re as f32  } }
impl Cast<f64> for c64 { fn cast(self) -> f64 { self.re as f64  } }
impl Cast<c32> for c64 { fn cast(self) -> c32 { c32::new(self.re as f32, self.im as f32) } }
impl Cast<c64> for c64 { fn cast(self) -> c64 { self } }

