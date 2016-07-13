#[allow(unused_imports)]
use array::NDArray;
#[allow(unused_imports)]
use backend;
#[allow(unused_imports)]
use backend::compute_backend::{SizedBuffer, ComputeBackend};

#[test]
#[cfg(feature = "opencl")]
fn opencl_enumerate_devices() {
    let platforms = backend::opencl::OpenCL::get_platform_list().unwrap();
    for platform in platforms {
        println!("{}", platform);
        let devices = backend::opencl::OpenCL::get_device_list(&platform).unwrap();
        for device in devices {
            println!("{}", device);
        }
    }
}

#[test]
#[cfg(feature = "opencl")]
fn opencl_init_and_finalize_devices() {
    let platforms = backend::opencl::OpenCL::get_platform_list().unwrap();
    for platform in platforms {
        println!("{}", platform);
        let mut devices = backend::opencl::OpenCL::get_device_list(&platform).unwrap();
        for device in &mut devices {
            device.init().unwrap();
            device.finalize().unwrap();
        }
    }
}

#[test]
#[cfg(feature = "opencl")]
fn opencl_backend_array() {
    let mut array1 = NDArray::<f32>::new(&[3,3], 0f32); 
    let mut array2 = NDArray::<f32>::new(&[3,3], 0f32); 
    for i in 0..3 {
        for j in 0..3 {
            array1[&[i,j]] = (i * 3 + j * 5) as f32;
        }
    }

    let platforms = backend::opencl::OpenCL::get_platform_list().unwrap();
    for platform in platforms {
        let mut devices = backend::opencl::OpenCL::get_device_list(&platform).unwrap();
        for device in &mut devices {
            device.init().unwrap();
            let id = device.create_array(array1.get_raw_size()).unwrap();       
            device.set_array(id, &array1).unwrap();
            device.get_array(id, &mut array2).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(array2[&[i,j]], (i * 3 + j * 5) as f32);
                }
            }
            device.delete_array(id).unwrap();
            device.finalize().unwrap();
        }
    }
}
