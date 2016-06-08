extern crate num;
extern crate byteorder;

use std::error::Error;
use std::fs::File;
use std::io::{Read,Write};
use std::iter::repeat;
use std::str::FromStr;
use std::string::FromUtf8Error;

use self::num::{FromPrimitive, ToPrimitive};
use self::byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use array::{NDArray, NDData};

const NUMPY_MAGIC : [u8;6] = [0x93u8, b'N', b'U', b'M', b'P', b'Y'];

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

pub enum Order {
    RowMajor,
    ColumnMajor
}

pub enum Endianess {
    LittleEndian,
    BigEndian,
}

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


impl NumpyFile {
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

    pub fn read_array<T : Clone + FromPrimitive>(&mut self) -> Result<NDArray<T>, String>  {
        let mut reader = match self.get_reader() {
            Ok(r) => r,
            Err(e) => return Err(e)
        };
        if let Err(e) = self.read_header(&mut reader) {
            return Err(e);
        }

        let readchain : Box<Fn(&mut File) -> Result<T, String>> = match self.dtype {
            DType::U8 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_u8() { 
                    Ok(v) => match T::from_u8(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from u8 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
            DType::U16 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_u16::<LittleEndian>() { 
                    Ok(v) => match T::from_u16(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from u16 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
            DType::U32 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_u32::<LittleEndian>() { 
                    Ok(v) => match T::from_u32(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from u32 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
            DType::U64 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_u64::<LittleEndian>() { 
                    Ok(v) => match T::from_u64(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from u64 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
            DType::I8 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_i8() { 
                    Ok(v) => match T::from_i8(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from i8 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
            DType::I16 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_i16::<LittleEndian>() { 
                    Ok(v) => match T::from_i16(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from i16 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
            DType::I32 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_i32::<LittleEndian>() { 
                    Ok(v) => match T::from_i32(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from i32 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
            DType::I64 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_i64::<LittleEndian>() { 
                    Ok(v) => match T::from_i64(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from i64 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
            DType::F32 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_f32::<LittleEndian>() { 
                    Ok(v) => match T::from_f32(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from f32 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
            DType::F64 => Box::new(|ref mut reader| -> Result<T, String> {
                match reader.read_f64::<LittleEndian>() { 
                    Ok(v) => match T::from_f64(v) {
                        Some(c) => Ok(c),
                        None => return Err(format!("Failed to convert value from f64 to T"))
                    },
                    Err(e) => return Err(e.description().to_string())
                }
            }),
        };

        let mut array = NDArray::<T>::new(&self.shape, T::from_u8(0).unwrap());
        let mut idx : Vec<usize> = repeat(0usize).take(self.shape.len()).collect();

        loop {
            array[&idx[..]] = match readchain(&mut reader) {
                Ok(v) => v,
                Err(e) => return Err(e)
            };
            match self.order {
                Order::ColumnMajor => {
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
                Order::RowMajor => {
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
}
