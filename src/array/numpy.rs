extern crate num;
extern crate byteorder;

use std::error::Error;
use std::fmt::Display;
use std::fs::File;
use std::io::{Read,Write};
use std::iter::repeat;
use std::str::FromStr;

use self::num::{FromPrimitive, ToPrimitive};
use self::byteorder::{ByteOrder, BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};

use array::{NDArray, NDData};
use array::ndindex::NDIndex;

const NUMPY_MAGIC : [u8;6] = [0x93u8, b'N', b'U', b'M', b'P', b'Y'];

/// Enumeration representing the supported numpy dtypes.
pub enum DType {
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
}

/// Enumeration representing the storage order.
pub enum Order {
    RowMajor,
    ColumnMajor
}

/// Enumeration representing the endianess.
pub enum Endianess {
    LittleEndian,
    BigEndian,
}

/// Structure representing a numpy array file (.npy).
pub struct NumpyFile {
    path : String,
    shape : Vec<usize>,
    pub dtype : DType,
    pub order : Order,
    pub endianess : Endianess,
}
    
fn extract_in_between(source : &str, start : &str, end : &str) -> Option<String> {
    let idx1 = match source.find(start) {
        Some(i) => i + start.len(),
        None => return None
    };
    let idx2 = match source[idx1..].find(end) {
        Some(i) => i + idx1,
        None => return None
    };
    return Some(source[idx1..idx2].to_string());
}

fn read_to_u8<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_u8() { 
        Ok(v) => match T::from_u8(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from u8 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn read_to_u16<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_u16::<B>() { 
        Ok(v) => match T::from_u16(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from u16 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn read_to_u32<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_u32::<B>() { 
        Ok(v) => match T::from_u32(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from u32 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn read_to_u64<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_u64::<B>() { 
        Ok(v) => match T::from_u64(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from u64 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn read_to_i8<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_i8() { 
        Ok(v) => match T::from_i8(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from i8 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn read_to_i16<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_i16::<B>() { 
        Ok(v) => match T::from_i16(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from i16 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn read_to_i32<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_i32::<B>() { 
        Ok(v) => match T::from_i32(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from i32 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn read_to_i64<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_i64::<B>() { 
        Ok(v) => match T::from_i64(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from i64 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn read_to_f32<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_f32::<B>() { 
        Ok(v) => match T::from_f32(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from f32 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn read_to_f64<T : FromPrimitive, B : ByteOrder>(reader : &mut File) -> Result<T, String> {
    match reader.read_f64::<B>() { 
        Ok(v) => match T::from_f64(v) {
            Some(c) => Ok(c),
            None => return Err(format!("Failed to convert value from f64 to T"))
        },
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_u8<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_u8(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to u8"))
    };
    match writer.write_u8(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_u16<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_u16(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to u16"))
    };
    match writer.write_u16::<B>(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_u32<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_u32(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to u32"))
    };
    match writer.write_u32::<B>(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_u64<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_u64(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to u64"))
    };
    match writer.write_u64::<B>(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_i8<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_i8(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to i8"))
    };
    match writer.write_i8(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_i16<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_i16(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to i16"))
    };
    match writer.write_i16::<B>(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_i32<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_i32(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to i32"))
    };
    match writer.write_i32::<B>(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_i64<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_i64(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to i64"))
    };
    match writer.write_i64::<B>(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_f32<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_f32(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to f32"))
    };
    match writer.write_f32::<B>(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

fn write_to_f64<T : ToPrimitive, B : ByteOrder>(writer : &mut File, v : T) -> Result<(), String> {
    let c = match T::to_f64(&v) {
        Some(c) => c,
        None => return Err(format!("Failed to convert value from T to f64"))
    };
    match writer.write_f64::<B>(c) {
        Ok(()) => return Ok(()),
        Err(e) => return Err(e.description().to_string())
    }
}

impl NumpyFile {

    /// Allocate a new NumpyFile structure with a given path. This function neither create nor open 
    /// the file specified by the path.
    pub fn new(path : &str) -> NumpyFile {
        NumpyFile {
            path : path.to_string(),
            shape : Vec::new(),
            dtype : DType::F32,
            order : Order::RowMajor,
            endianess : Endianess::LittleEndian,
        }
    }
    
    fn get_reader(&self) -> Result<File, String> {
        match File::open(&self.path[..]) {
            Ok(r) => Ok(r),
            Err(e) => Err(e.description().to_string()),
        }
    }

    fn get_writer(&self) -> Result<File, String> {
        match File::create(&self.path[..]) {
            Ok(r) => Ok(r),
            Err(e) => Err(e.description().to_string()),
        }
    }

    /// Open the Numpy file for reading and parse the header, storing the results in the dtype, 
    /// order and endianess fields.
    /// In case of failure, returns the error as a string.
    #[allow(unused_assignments)]
    pub fn read_header(&mut self, file : &mut File) -> Result<(),String> {
        let mut magic = [0u8;6];
        let mut version = [0u8;2];
        let mut header_size = 0u32;

        if let Err(e) = file.read_exact(&mut magic) {
            return Err(e.description().to_string());
        }
        if magic != NUMPY_MAGIC {
            return Err(format!("File {} does not have a valid numpy magic", self.path));
        }

        if let Err(e) = file.read_exact(&mut version) {
            return Err(e.description().to_string());
        }

        if version[0] == 0x1 {
            let mut buf = [0u8;2];
            if let Err(e) = file.read_exact(&mut buf) {
                return Err(e.description().to_string());
            }
            header_size = (buf[0] as u32) + ((buf[1] as u32) << 8);
        }
        else if version[0] == 0x2 {
            let mut buf = [0u8;4];
            if let Err(e) = file.read_exact(&mut buf) {
                return Err(e.description().to_string());
            }
            header_size = (buf[0] as u32) + ((buf[1] as u32) << 8) + ((buf[2] as u32) << 16) + ((buf[3] as u32) << 24);
        }
        else {
            return Err(format!("Numpy file major version number {} not suppored", version[0]));
        }

        let mut header_raw : Vec<u8>= repeat(0u8).take(header_size as usize).collect();
        if let Err(e) = file.read_exact(&mut header_raw[..]) {
            return Err(e.description().to_string());
        }
        let header = match String::from_utf8(header_raw) {
            Ok(s) => s,
            Err(e) => return Err(e.description().to_string())
        };

        // {'descr': '<i8', 'fortran_order': False, 'shape': (5,), }
        let descr = match extract_in_between(&header[..], "'descr': '", "',") {
            Some(s) => s,
            None => return Err(format!("descr not present in numpy header : {}", header))
        };
        let fortran_order = match extract_in_between(&header[..], "'fortran_order': ", ",") {
            Some(s) => s,
            None => return Err(format!("fortran_order not present in numpy header : {}", header))
        };
        let shape = match extract_in_between(&header[..], "'shape': (", ")") {
            Some(s) => s,
            None => return Err(format!("shape not present in numpy header : {}", header))
        };

        if descr.len() < 1 {
            return Err(format!("descr empty"));
        }

        self.endianess = match &descr[0..1] {
            "<" | "|" => Endianess::LittleEndian,
            ">" => Endianess::BigEndian,
            _ => return Err(format!("Failed to parse descr endianess: {}", descr))
        };

        self.dtype = match &descr[1..] {
            "u1" => DType::U8,
            "u2" => DType::U16,
            "u4" => DType::U32,
            "u8" => DType::U64,
            "i1" => DType::I8,
            "i2" => DType::I16,
            "i4" => DType::I32,
            "i8" => DType::I64,
            "f4" => DType::F32,
            "f8" => DType::F64,
            _ => return Err(format!("Unsuported descr type: {}", descr))
        };
        
        self.order = match &fortran_order[..] {
            "False" => Order::RowMajor,
            "True" => Order::ColumnMajor,
            _ => return Err(format!("Invalid fortran_order: {}", fortran_order))
        };

        self.shape.clear();
        for s in shape.split(',') {
            let s = s.trim();
            if s.len() > 0 {
                match usize::from_str(s) {
                    Ok(u) => self.shape.push(u),
                    Err(e) => return Err(e.description().to_string())
                }
            }
        }

        return Ok(());
    }
    
    fn write_header(&mut self, file : &mut File) -> Result<(),String> {
        let mut header = Vec::<u8>::new();

        if let Err(e) = file.write_all(&NUMPY_MAGIC) {
            return Err(e.description().to_string());
        }
        if let Err(e) = file.write_all(&[1u8,0u8]) {
            return Err(e.description().to_string());
        }

        header.extend_from_slice("{'descr': '".as_bytes());
        header.push(
            match self.endianess {
                Endianess::BigEndian => b'>',
                Endianess::LittleEndian => b'<',
            }
        );
        header.extend_from_slice(
            match self.dtype {
                DType::U8  => "u1".as_bytes(),
                DType::U16 => "u2".as_bytes(),
                DType::U32 => "u4".as_bytes(),
                DType::U64 => "u8".as_bytes(),
                DType::I8  => "i1".as_bytes(),
                DType::I16 => "i2".as_bytes(),
                DType::I32 => "i4".as_bytes(),
                DType::I64 => "i8".as_bytes(),
                DType::F32 => "f4".as_bytes(),
                DType::F64 => "f8".as_bytes(),
           }
        );
        header.extend_from_slice("', 'fortran_order': ".as_bytes());
        header.extend_from_slice(
            match self.order {
                Order::RowMajor => "False".as_bytes(),
                Order::ColumnMajor => "True".as_bytes(),
            }
        );
        header.extend_from_slice(", 'shape': (".as_bytes());
        for s in & self.shape {
            header.extend_from_slice(&format!("{}, ", s).into_bytes()[..]);
        }
        header.extend_from_slice("), }".as_bytes());

        let pad = (header.len() + 11).wrapping_neg() % 16;
        header.append(&mut repeat(b' ').take(pad).collect());
        header.push(b'\n');

        if let Err(e) = file.write_u16::<LittleEndian>(header.len() as u16) {
            return Err(e.description().to_string());
        }
        if let Err(e) = file.write_all(&header[..]) {
            return Err(e.description().to_string());
        }

        return Ok(());
    }

    /// Open the Numpy file for reading and read the entire numpy array as a NDArray<T>. This 
    /// function operates its own type convertion from the dtype to the type T.
    /// In case of failure, returns the error as a string.
    pub fn read_array<T : Clone + FromPrimitive + Display>(&mut self) -> Result<NDArray<T>, String>  {
        let mut reader = match self.get_reader() {
            Ok(r) => r,
            Err(e) => return Err(e)
        };
        if let Err(e) = self.read_header(&mut reader) {
            return Err(e);
        }

        let readchain = match self.dtype {
            DType::U8 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_u8::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_u8::<T, LittleEndian>,
                }
            },
            DType::U16 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_u16::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_u16::<T, LittleEndian>,
                }
            },
            DType::U32 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_u32::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_u32::<T, LittleEndian>,
                }
            },
            DType::U64 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_u64::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_u64::<T, LittleEndian>,
                }
            },
            DType::I8 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_i8::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_i8::<T, LittleEndian>,
                }
            },
            DType::I16 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_i16::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_i16::<T, LittleEndian>,
                }
            },
            DType::I32 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_i32::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_i32::<T, LittleEndian>,
                }
            },
            DType::I64 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_i64::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_i64::<T, LittleEndian>,
                }
            },
            DType::F32 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_f32::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_f32::<T, LittleEndian>,
                }
            },
            DType::F64 => {
                match self.endianess {
                    Endianess::BigEndian    => read_to_f64::<T, BigEndian>,
                    Endianess::LittleEndian => read_to_f64::<T, LittleEndian>,
                }
            },
        };

        let mut array = NDArray::<T>::new(&self.shape, T::from_u8(0).unwrap());
        let mut idx : Vec<usize> = repeat(0usize).take(self.shape.len()).collect();

        loop {
            array[&idx[..]] = match readchain(&mut reader) {
                Ok(v) => v,
                Err(e) => return Err(e)
            };
            match self.order {
                Order::RowMajor => {
                    let mut i = idx.len();
                    while i > 0 {
                        idx[i-1] += 1;
                        if idx[i-1] >= self.shape[i-1] {
                            idx[i-1] = 0;
                            i -= 1;
                        }
                        else {
                            break;
                        }
                    }
                    if i == 0 {
                        break;
                    }
                },
                Order::ColumnMajor => {
                    let mut i = 0;
                    while i < idx.len() {
                        idx[i] += 1;
                        if idx[i] >= self.shape[i] {
                            idx[i] = 0;
                            i += 1;
                        }
                        else {
                            break;
                        }
                    }
                    if i == idx.len() {
                        break;
                    }
                }
            }
        }

        return Ok(array);
    }

    /// Open (or create) the Numpy file for writing and write the entire NDData<T> in it. This 
    /// function operates its own type convertion from the type T to the dtype. It is thus 
    /// important to specify the desired dtype in the NumpyFile structure.
    /// In case of failure, returns the error as a string.
    pub fn write_array<T : Clone + ToPrimitive + Display>(&mut self, array : &NDData<T>) -> Result<(), String>  {
        let mut writer = match self.get_writer() {
            Ok(w) => w,
            Err(e) => return Err(e)
        };
        self.shape = array.shape().to_vec();
        if let Err(e) = self.write_header(&mut writer) {
            return Err(e);
        }

        let writechain = match self.dtype {
            DType::U8 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_u8::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_u8::<T, LittleEndian>,
                }
            },
            DType::U16 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_u16::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_u16::<T, LittleEndian>,
                }
            },
            DType::U32 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_u32::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_u32::<T, LittleEndian>,
                }
            },
            DType::U64 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_u64::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_u64::<T, LittleEndian>,
                }
            },
            DType::I8 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_i8::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_i8::<T, LittleEndian>,
                }
            },
            DType::I16 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_i16::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_i16::<T, LittleEndian>,
                }
            },
            DType::I32 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_i32::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_i32::<T, LittleEndian>,
                }
            },
            DType::I64 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_i64::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_i64::<T, LittleEndian>,
                }
            },
            DType::F32 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_f32::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_f32::<T, LittleEndian>,
                }
            },
            DType::F64 => {
                match self.endianess {
                    Endianess::BigEndian    => write_to_f64::<T, BigEndian>,
                    Endianess::LittleEndian => write_to_f64::<T, LittleEndian>,
                }
            },
        };

        let mut idx : Vec<usize> = repeat(0usize).take(array.dim()).collect();

        loop {
            if let Err(e) = writechain(&mut writer, array.idx(&idx[..]).clone()) {
                return Err(e)
            }
            match self.order {
                Order::RowMajor => {
                    idx.inc_ro(array.shape())
                },
                Order::ColumnMajor => {
                    idx.inc_co(array.shape())
                }
            }
            if idx.is_zero() {
                break;
            }
        }

        return Ok(());
    }
}
