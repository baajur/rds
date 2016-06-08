extern crate csv;

use std::clone::Clone;
use std::error::Error;
use std::fmt::Display;
use std::fs::File;
//use std::marker::Sized;
//use std::ops::Index;
use std::str::FromStr;

use array::{NDArray, NDData};

pub struct CSVFile {
    path : String,
    pub header : bool,
    pub flexible : bool,
    pub delimiter : u8,
    pub quote : u8,
}

impl CSVFile {
    pub fn new(path : &str) -> CSVFile {
        CSVFile {
            path : path.to_string(),
            header : false,
            flexible : false,
            delimiter : b',',
            quote : b'"',
        }
    }

    fn get_reader(&self) -> Result<csv::Reader<File>, String> {
        match csv::Reader::from_file(&self.path[..]) {
            Ok(r) => {
                return Ok(r.has_headers(self.header)
                           .flexible(self.flexible)
                           .delimiter(self.delimiter)
                           .quote(self.quote));
            }
            Err(e) => {
                Err(e.description().to_string())
            }
        }
    }

    #[allow(unused_mut)] 
    fn get_writer(&mut self) -> Result<csv::Writer<File>, String> {
        match csv::Writer::from_file(&self.path[..]) {
            Ok(w) => {
                return Ok(w.flexible(self.flexible)
                           .delimiter(self.delimiter)
                           .quote(self.quote));
            }
            Err(e) => {
                Err(e.description().to_string())
            }
        }
    }

    pub fn read_row<T : FromStr + Clone>(&self, row_idx : usize) -> Result<NDArray<T>, String> {
        let mut data = Vec::<T>::new();
        let mut shape = [0usize;1];
        let mut reader = match self.get_reader() {
            Ok(r) => r,
            Err(e) => return Err(e),
        };

        match reader.records().nth(row_idx) {
            Some(Ok(record)) => {
                shape[0] = record.len();
                for item in record {
                    match T::from_str(&item[..]) {
                        Ok(value) => {
                            data.push(value);
                        },
                        Err(_) => {
                            return Err(format!("Failed to parse value '{:}' at column {:} of row {:}", item, shape[1], row_idx));
                        }
                    }
                }
            },
            Some(Err(e)) => {
                return Err(e.description().to_string());
            },
            None => {
                return Err(format!("Row {:} not found", row_idx));
            }
        }
        return Ok(NDArray::from_slice(&shape[..], &data[..]));
    }

    pub fn read_column<T : FromStr + Clone>(&self, column_idx : usize) -> Result<NDArray<T>, String> {
        let mut data = Vec::<T>::new();
        let mut shape = [0usize;1];
        let mut reader = match self.get_reader() {
            Ok(r) => r,
            Err(e) => return Err(e),
        };

        for record in reader.records() {
            match record {
                Ok(record) => {
                    shape[0] += 1;
                    if record.len() > column_idx {
                        match T::from_str(&record[column_idx][..]) {
                            Ok(value) => {
                                data.push(value);
                            },
                            Err(_) => {
                                return Err(format!("Failed to parse value '{:}' at row {:} of column {:}", record[column_idx], shape[1], column_idx));
                            }
                        }
                    }
                    else {
                        return Err(format!("Column {:} not found in row {:}", column_idx, shape[1]));
                    }
                },
                Err(e) => {
                    return Err(e.description().to_string());
                }
            }
        }
        return Ok(NDArray::from_slice(&shape[..], &data[..]));
    }

    pub fn read_2darray<T : FromStr + Clone>(&self) -> Result<NDArray<T>, String> {
        let mut data = Vec::<T>::new();
        let mut shape = [0usize;2];
        let mut reader = match self.get_reader() {
            Ok(r) => r,
            Err(e) => return Err(e),
        };

        for record in reader.records() {
            match record {
                Ok(record) => {
                    shape[0] += 1;
                    shape[1] = record.len();
                    for item in record {
                        match T::from_str(&item[..]) {
                            Ok(value) => {
                                data.push(value);
                            },
                            Err(_) => {
                                return Err(format!("Failed to parse value '{:}' at record {:}", item, shape[1]));
                            }
                        }
                    }
                },
                Err(e) => {
                    return Err(e.description().to_string());
                }
            }
        }
        return Ok(NDArray::from_slice(&shape[..], &data[..]));
    }

    pub fn write_data<T : Display>(&mut self, data : &NDData<T>) -> Result<(), String>  {
        assert!(data.dim() <= 2);

        let mut writer = match self.get_writer() {
            Ok(w) => w,
            Err(e) => return Err(e),
        };

        if data.dim() == 2 {
            for i in 0..data.shape()[0] {
                let mut record = Vec::<String>::new();
                for j in 0..data.shape()[1] {
                    let idx = i * data.strides()[0] + j * data.strides()[1];
                    record.push(format!("{}", data.get_data()[idx]))
                }
                if let Err(e) = writer.write(record.into_iter()) {
                    return Err(e.description().to_string());
                }
            }
        }
        else if data.dim() == 1 {
            let mut record = Vec::<String>::new();
            for i in 0..data.shape()[1] {
                let idx = i * data.strides()[0];
                record.push(format!("{}", data.get_data()[idx]))
            }
            if let Err(e) = writer.write(record.into_iter()) {
                return Err(e.description().to_string());
            }
        }
        else if data.dim() == 0 {
            let mut record = Vec::<String>::new();
            record.push(format!("{}", data.get_data()[0]));
            if let Err(e) = writer.write(record.into_iter()) {
                return Err(e.description().to_string());
            }
        }

        return Ok(());
    }
}
