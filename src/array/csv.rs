extern crate csv;

use std::clone::Clone;
use std::error::Error;
use std::io::{Read,Write};
use std::fmt::Display;
use std::str::FromStr;

use array::{Array, NDData};

pub fn array1d_from_csv_row<T : FromStr + Clone>(path : &str, row_idx : usize) -> Result<Array<T>, String> {
    match csv::Reader::from_file(path) {
        Ok(r) => array1d_from_csvreader_row(r.flexible(true), row_idx),
        Err(e) => Err(e.description().to_string())
    }
}

pub fn array1d_from_csvreader_row<R : Read, T : FromStr + Clone>(mut reader : csv::Reader<R>, row_idx : usize) -> Result<Array<T>, String> {
    let mut data = Vec::<T>::new();
    let mut shape = [0usize;1];
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
    return Ok(Array::from_slice(&shape[..], &data[..]));
}

pub fn array1d_from_csv_column<T : FromStr + Clone>(path : &str, column_idx : usize) -> Result<Array<T>, String> {
    match csv::Reader::from_file(path) {
        Ok(r) => array1d_from_csvreader_row(r, column_idx),
        Err(e) => Err(e.description().to_string())
    }
}

pub fn array1d_from_csvreader_column<R : Read, T : FromStr + Clone>(mut reader : csv::Reader<R>, column_idx : usize) -> Result<Array<T>, String> {
    let mut data = Vec::<T>::new();
    let mut shape = [0usize;1];
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
    return Ok(Array::from_slice(&shape[..], &data[..]));
}

pub fn array2d_from_csv<T : FromStr + Clone>(path : &str) -> Result<Array<T>, String> {
    match csv::Reader::from_file(path) {
        Ok(r) => array2d_from_csvreader(r),
        Err(e) => Err(e.description().to_string())
    }
}

pub fn array2d_from_csvreader<R : Read, T : FromStr + Clone>(mut reader : csv::Reader<R>) -> Result<Array<T>, String> {
    let mut data = Vec::<T>::new();
    let mut shape = [0usize;2];
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
    return Ok(Array::from_slice(&shape[..], &data[..]));
}

pub fn array2d_to_csv<T : Display>(path : &str, array : &Array<T>) -> Result<(), String> {
    match csv::Writer::from_file(path) {
        Ok(w) => array2d_to_csvwriter(w, array),
        Err(e) => Err(e.description().to_string())
    }
}

pub fn array2d_to_csvwriter<W : Write, T : Display>(mut writer : csv::Writer<W>, array : &Array<T>) -> Result<(), String> {
    assert!(array.dim() == 2);
    for i in 0..array.shape()[0] {
        let mut record = Vec::<String>::new();
        for j in 0..array.shape()[1] {
            record.push(format!("{}", array[&[i, j][..]]))
        }
        if let Err(e) = writer.write(record.into_iter()) {
            return Err(e.description().to_string());
        }
    }
    return Ok(());
}
