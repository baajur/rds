use backend;
use backend::compute_backend::ComputeBackend;

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
