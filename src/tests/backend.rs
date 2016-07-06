use backend::opencl::{OpenCL, CLPlatform};

#[test]
fn opencl_enumerate_devices() {
    let platforms = OpenCL::get_platform_list().unwrap();
    for platform in platforms {
        println!("{}", platform);
        let devices = OpenCL::get_device_list(&platform).unwrap();
        for device in devices {
            println!("{}", device);
        }
    }
}
