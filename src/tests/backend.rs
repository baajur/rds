use backend;

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
