use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::iter::repeat;
use std::slice;
use std::str;
use std::str::FromStr;

use types::cast::Cast;
use array::NDArray;

/// Structure representing an EDFFile.
pub struct EDFFile {
    path : String,
}

impl EDFFile {
    
    /// Allocate a new EDFFile structure with a given path. This function neither create nor open 
    /// the file specified by the path.
    pub fn new(path : &str) -> EDFFile{
        EDFFile {
            path : path.to_string(),
        }
    }

    /// Read signal number id (id starting at 0) from the EDFFile into an one dimension NDArray.
    pub fn read_signal<T : Copy>(&self, id : usize) -> Result<NDArray<T>,String> where i16 : Cast<T>{
        let mut header = [0u8;256];
        let mut signal = Vec::<T>::new();

        let mut reader = match File::open(&self.path[..]) {
            Ok(r) => r,
            Err(e) => return Err(e.description().to_string())
        };

        // Read main header and extract the number of signals
        if let Err(e) = reader.read_exact(&mut header) {
            return Err(e.description().to_string());
        }
        let num_signal = match str::from_utf8(&header[252..256]) {
            Ok(s) => match usize::from_str(s.trim()) {
                Ok(v) =>v,
                Err(e) => return Err(e.description().to_string())
            },
            Err(e) => return Err(e.description().to_string())
        };
        if id >= num_signal {
            return Err(format!("EDFFile::read_signal(): The signal id is greater than the number of signal of the file {} ({} >= {})", self.path, id, num_signal));
        }
        let num_record = match str::from_utf8(&header[236..244]) {
            Ok(s) => match usize::from_str(s.trim()) {
                Ok(v) =>v,
                Err(e) => return Err(e.description().to_string())
            },
            Err(e) => return Err(e.description().to_string())
        };

        // Read each signal header and extract their number of samples
        let mut num_samples = Vec::<usize>::new();
        if let Err(e) = reader.seek(SeekFrom::Current(num_signal as i64 * 216)) {
            return Err(e.description().to_string());
        }
        for _ in 0..num_signal {
            let mut buffer = [0u8;8];
            if let Err(e) = reader.read_exact(&mut buffer) {
                return Err(e.description().to_string());
            }
            num_samples.push(match str::from_utf8(&buffer) {
                Ok(s) => match usize::from_str(s.trim()) {
                    Ok(v) =>v,
                    Err(e) => return Err(e.description().to_string())
                },
                Err(e) => return Err(e.description().to_string())
            });
        }

        if let Err(e) = reader.seek(SeekFrom::Current(num_signal as i64 * 32)) {
            return Err(e.description().to_string());
        }
        let mut data_record : Vec<u8> = repeat(0u8).take(num_samples[id] * 2).collect();
        for _ in 0..num_record {
            for i in 0..num_signal {
                if id == i {
                    if let Err(e) = reader.read_exact(&mut data_record[..]) {
                        return Err(e.description().to_string());
                    }
                    unsafe {
                        let transmuted : &[i16]= slice::from_raw_parts(data_record.as_ptr() as *const i16, num_samples[id]);
                        for v in transmuted {
                            signal.push(Cast::<T>::cast(*v));
                        }
                    }
                }
                else {
                    if let Err(e) = reader.seek(SeekFrom::Current(num_samples[i] as i64 * 2)) {
                        return Err(e.description().to_string());
                    }
                }
            }
        }

        return Ok(NDArray::<T>::from_slice(&[signal.len()], &signal[..]));
    }
}
