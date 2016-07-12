extern crate libc;

use std::iter::repeat;
use std::fmt;
use std::ptr;
use std::mem::{transmute, size_of};

use backend::compute_backend::{SizedBuffer, SizedBufferMut, ComputeBackend, ArrayRegistry};

pub type CLPlatformId = *const libc::c_void;
pub type CLDeviceId = *const libc::c_void;
pub type CLContext = *const libc::c_void;
pub type CLCommandQueue = *const libc::c_void;
pub type CLMem = *const libc::c_void;

const CL_PLATFORM_PROFILE : u32 = 0x0900u32;
const CL_PLATFORM_VERSION : u32 = 0x0901u32;
const CL_PLATFORM_NAME : u32 = 0x0902u32;
const CL_PLATFORM_VENDOR : u32 = 0x0903u32;
const CL_PLATFORM_EXTENSIONS : u32 = 0x0904u32;

const CL_DEVICE_NAME : u32 = 0x102Bu32;
const CL_DEVICE_VENDOR : u32 = 0x102Cu32;
const CL_DEVICE_PROFILE : u32 = 0x102Eu32;
const CL_DEVICE_VERSION : u32 = 0x102Fu32;
const CL_DEVICE_EXTENSIONS : u32 = 0x1030u32;
const CL_DEVICE_MAX_CLOCK_FREQUENCY : u32 = 0x100Cu32;
const CL_DEVICE_MAX_COMPUTE_UNITS : u32 = 0x1002u32;
const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS : u32 = 0x1003u32;
const CL_DEVICE_MAX_WORK_GROUP_SIZE : u32 = 0x1004u32;
const CL_DEVICE_MAX_WORK_ITEM_SIZES : u32 = 0x1005u32;
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT : u32 = 0x100Au32;
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE : u32 = 0x100Bu32;
const CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE : u32 = 0x101Du32;
const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE : u32 = 0x101Eu32;
const CL_DEVICE_GLOBAL_MEM_SIZE : u32 = 0x101Fu32;
const CL_DEVICE_LOCAL_MEM_SIZE : u32 = 0x1023u32;

const CL_SUCCESS : i32 = 0i32;

const CL_DEVICE_TYPE_ALL : u32 = 0xffffffffu32;

const CL_MEM_READ_WRITE : u32 = 1u32;

#[link(name = "OpenCL")]
extern {
    fn clGetPlatformIDs(num_entries : u32, platforms : *mut CLPlatformId, num_platforms : *mut u32) -> i32;
    fn clGetPlatformInfo(platform : CLPlatformId, param_name : u32, param_value_size : usize, param_value : *mut libc::c_void, param_value_size_ret : *mut usize) -> i32;
    fn clGetDeviceIDs(platform : CLPlatformId, device_type : u32, num_entries : u32, devices : *mut CLDeviceId, num_devices : *mut u32) -> i32;
    fn clGetDeviceInfo(device : CLDeviceId, param_name : u32, param_value_size : usize, param_value : *mut libc::c_void, param_value_size_ret : *mut usize) -> i32;
    fn clCreateContext(properties : *const libc::c_void, num_devices : u32, devices : *const CLDeviceId, pfn_notify : *const libc::c_void, user_data : *mut libc::c_void, errcode_ret : *mut i32) -> CLContext;
    fn clCreateCommandQueue(context : CLContext, device : CLDeviceId, properties : u32, errcode_ret : *mut i32) -> CLCommandQueue;
    fn clReleaseContext(context : CLContext) -> i32;
    fn clReleaseCommandQueue(command_queue : CLCommandQueue) -> i32;
    fn clCreateBuffer(context : CLContext, flags : u32, size : usize, host_ptr : *const libc::c_void, errcode_ret : *mut i32) -> CLMem;
    fn clReleaseMemObject(memobj : CLMem) -> i32;
    fn clEnqueueReadBuffer(command_queue : CLCommandQueue, buffer : CLMem, blocking_read : u32, offset : usize, size : usize, ptr : *const libc::c_void, num_events_in_wait_list : u32, event_wait_list : *const libc::c_void, event : *const libc::c_void) -> i32;
    fn clEnqueueWriteBuffer(command_queue : CLCommandQueue, buffer : CLMem, blocking_write : u32, offset : usize, size : usize, ptr : *const libc::c_void, num_events_in_wait_list : u32, event_wait_list : *const libc::c_void, event : *const libc::c_void) -> i32;
}

pub struct CLPlatform {
    id : CLPlatformId,
    name : String,
    vendor : String,
    version : String,
    profile : String,
    extensions : String
}

pub struct CLDevice {
    id : CLDeviceId,
    context : Option<CLContext>,
    command_queue : Option<CLCommandQueue>,
    array_registry : ArrayRegistry<CLMem>,
    name : String,
    vendor : String,
    version : String,
    profile : String,
    extensions : String,
    max_clock_frequency : u32,
    max_compute_units : u32,
    max_work_item_dim : u32,
    max_work_item_size : Vec<usize>,
    max_work_group_size : usize,
    preferred_float_vec_width : u32,
    preferred_double_vec_width : u32,
    global_mem_cacheline_size : u64,
    global_mem_cache_size : u64,
    global_mem_size : u64,
    local_mem_size : u64,
}

pub struct OpenCL {
}

impl CLPlatform {
    fn get_info(id : CLPlatformId, param : u32) -> Result<String, i32> {
        let mut size = 0usize;
        let retcode = unsafe { clGetPlatformInfo(id, param, 0, ptr::null_mut(), &mut size) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }

        let mut buffer : Vec<u8> = repeat(0u8).take(size).collect();
        let retcode = unsafe { clGetPlatformInfo(id, param, size, buffer.as_mut_ptr() as *mut libc::c_void, ptr::null_mut()) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }

        return Ok(String::from_utf8(buffer).unwrap());
    }

    pub fn new(id : CLPlatformId) -> Result<CLPlatform, i32> {
        Ok(CLPlatform {
            id : id,
            name : try!(CLPlatform::get_info(id, CL_PLATFORM_NAME)),
            vendor: try!(CLPlatform::get_info(id, CL_PLATFORM_VENDOR)),
            version: try!(CLPlatform::get_info(id, CL_PLATFORM_VERSION)),
            profile: try!(CLPlatform::get_info(id, CL_PLATFORM_PROFILE)),
            extensions: try!(CLPlatform::get_info(id, CL_PLATFORM_EXTENSIONS)),
        })
    }

    pub fn get_id(&self) -> CLPlatformId { self.id }

    pub fn get_name(&self) -> &str { &self.name[..] }

    pub fn get_vendor(&self) -> &str { &self.vendor[..] }

    pub fn get_version(&self) -> &str { &self.version[..] }

    pub fn get_profile(&self) -> &str { &self.profile[..] }

    pub fn get_extensions(&self) -> &str { &self.extensions[..] }
}

impl CLDevice {
    fn get_string_info(id : CLDeviceId, param : u32) -> Result<String, i32> {
        let mut size = 0usize;
        let retcode = unsafe { clGetDeviceInfo(id, param, 0, ptr::null_mut(), &mut size) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }

        let mut buffer : Vec<u8> = repeat(0u8).take(size).collect();
        let retcode = unsafe { clGetDeviceInfo(id, param, size, buffer.as_mut_ptr() as *mut libc::c_void, ptr::null_mut()) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }

        return Ok(String::from_utf8(buffer).unwrap());
    }

    fn get_uint_info(id : CLDeviceId, param : u32) -> Result<u32, i32> {
        let mut v = 0u32;
        let retcode = unsafe { clGetDeviceInfo(id, param, size_of::<u32>(), transmute(&mut v), ptr::null_mut()) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }
        return Ok(v);
    }

    fn get_ulong_info(id : CLDeviceId, param : u32) -> Result<u64, i32> {
        let mut v = 0u64;
        let retcode = unsafe { clGetDeviceInfo(id, param, size_of::<u64>(), transmute(&mut v), ptr::null_mut()) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }
        return Ok(v);
    }

    fn get_size_t_info(id : CLDeviceId, param : u32) -> Result<usize, i32> {
        let mut v = 0usize;
        let retcode = unsafe { clGetDeviceInfo(id, param, size_of::<usize>(), transmute(&mut v), ptr::null_mut()) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }
        return Ok(v);
    }

    fn get_size_t_vec_info(id : CLDeviceId, param : u32, n : usize) -> Result<Vec<usize>, i32> {
        let mut v : Vec<usize> = repeat(0usize).take(n).collect();
        let retcode = unsafe { clGetDeviceInfo(id, param, n * size_of::<usize>(), transmute(v.as_mut_ptr()), ptr::null_mut()) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }
        return Ok(v);
    }

    pub fn new(id : CLDeviceId) -> Result<CLDevice, i32> {
        let dim = try!(CLDevice::get_uint_info(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
        Ok(CLDevice {
            id : id,
            context : None,
            command_queue : None,
            array_registry : ArrayRegistry::<CLMem>::new(),
            name : try!(CLDevice::get_string_info(id, CL_DEVICE_NAME)),
            vendor : try!(CLDevice::get_string_info(id, CL_DEVICE_VENDOR)),
            version : try!(CLDevice::get_string_info(id, CL_DEVICE_VERSION)),
            profile : try!(CLDevice::get_string_info(id, CL_DEVICE_PROFILE)),
            extensions : try!(CLDevice::get_string_info(id, CL_DEVICE_EXTENSIONS)),
            max_clock_frequency : try!(CLDevice::get_uint_info(id, CL_DEVICE_MAX_CLOCK_FREQUENCY)),
            max_compute_units : try!(CLDevice::get_uint_info(id, CL_DEVICE_MAX_COMPUTE_UNITS)),
            max_work_item_dim : dim,
            max_work_item_size : try!(CLDevice::get_size_t_vec_info(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, dim as usize)),
            max_work_group_size : try!(CLDevice::get_size_t_info(id, CL_DEVICE_MAX_WORK_GROUP_SIZE)),
            preferred_float_vec_width : try!(CLDevice::get_uint_info(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)),
            preferred_double_vec_width : try!(CLDevice::get_uint_info(id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE)),
            global_mem_cacheline_size : try!(CLDevice::get_ulong_info(id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE)),
            global_mem_cache_size : try!(CLDevice::get_ulong_info(id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)),
            global_mem_size : try!(CLDevice::get_ulong_info(id, CL_DEVICE_GLOBAL_MEM_SIZE)),
            local_mem_size : try!(CLDevice::get_ulong_info(id, CL_DEVICE_LOCAL_MEM_SIZE)),

        })
    }

    pub fn get_id(&self) -> CLPlatformId { self.id }

    pub fn get_name(&self) -> &str { &self.name[..] }

    pub fn get_vendor(&self) -> &str { &self.vendor[..] }

    pub fn get_version(&self) -> &str { &self.version[..] }

    pub fn get_profile(&self) -> &str { &self.profile[..] }

    pub fn get_extensions(&self) -> &str { &self.extensions[..] }

    pub fn get_max_clock_frequency(&self) -> u32 { self.max_clock_frequency }

    pub fn get_max_compute_units(&self) -> u32 { self.max_compute_units }

    pub fn get_max_work_item_dim(&self) -> u32 { self.max_work_item_dim }

    pub fn get_max_work_item_size(&self) -> &[usize] { &self.max_work_item_size[..] }

    pub fn get_max_work_group_size(&self) -> usize { self.max_work_group_size }

    pub fn get_preferred_float_vec_width(&self) -> u32 { self.preferred_float_vec_width }

    pub fn get_preferred_double_vec_width(&self) -> u32 { self.preferred_double_vec_width }

    pub fn get_global_mem_cacheline_size(&self) -> u64 { self.global_mem_cacheline_size }

    pub fn get_global_mem_cache_size(&self) -> u64 { self.global_mem_cache_size }

    pub fn get_global_mem_size(&self) -> u64 { self.global_mem_size }

    pub fn get_local_mem_size(&self) -> u64 { self.local_mem_size }

}

impl fmt::Display for CLPlatform {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "OpenCL Platform {}\n", self.get_name()));
        try!(write!(f, "\tBy {}\n", self.get_vendor()));
        try!(write!(f, "\tVersion {}\n", self.get_version()));
        try!(write!(f, "\tWith profile {}\n", self.get_profile()));
        try!(write!(f, "\tImplementing extensions {}\n", self.get_extensions()));
        return Ok(());
    }
}

impl fmt::Display for CLDevice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "OpenCL Device {}\n", self.get_name()));
        try!(write!(f, "\tBy {}\n", self.get_vendor()));
        try!(write!(f, "\tVersion {}\n", self.get_version()));
        try!(write!(f, "\tWith profile {}\n", self.get_profile()));
        try!(write!(f, "\tImplementing extensions {}\n", self.get_extensions()));
        try!(write!(f, "\tMax clock frequency: {} MHz\n", self.get_max_clock_frequency()));
        try!(write!(f, "\tMax compute units: {}\n", self.get_max_compute_units()));
        try!(write!(f, "\tMax work item dimensions: {}\n", self.get_max_work_item_dim()));
        try!(write!(f, "\tMax work item size: {:?}\n", self.get_max_work_item_size()));
        try!(write!(f, "\tMax preferred float vector width: {}\n", self.get_preferred_float_vec_width()));
        try!(write!(f, "\tMax preferred double vector width: {}\n", self.get_preferred_double_vec_width()));
        try!(write!(f, "\tGlobal memory cacheline size: {} B\n", self.get_global_mem_cacheline_size()));
        try!(write!(f, "\tGlobal memory cache size: {} B\n", self.get_global_mem_cache_size()));
        try!(write!(f, "\tGlobal memory size: {} B\n", self.get_global_mem_size()));
        try!(write!(f, "\tLocal memory size: {} B\n", self.get_local_mem_size()));
        return Ok(());
    }
}

impl ComputeBackend for CLDevice {

    fn init(&mut self) -> Result<(),String> {
        let mut retcode = CL_SUCCESS;

        let context = unsafe { clCreateContext(ptr::null(), 1, &self.id, ptr::null(), ptr::null_mut(), &mut retcode) };
        if retcode != CL_SUCCESS {
            return Err(format!("Failed to initialize OpenCL context for device {}", self.get_name()));
        }
        self.context = Some(context);

        let command_queue = unsafe { clCreateCommandQueue(context, self.id, 0, &mut retcode) };
        if retcode != CL_SUCCESS {
            return Err(format!("Failed to initialize OpenCL command queue for device {}", self.get_name()));
        }
        self.command_queue = Some(command_queue);

        return Ok(());
    }

    fn create_array(&mut self, size : usize) -> Result<u32,String> {
        if let Some(context) = self.context {
            let mut retcode = CL_SUCCESS;
            let array = unsafe {clCreateBuffer(context, CL_MEM_READ_WRITE, size, ptr::null(), &mut retcode) };
            if retcode != CL_SUCCESS {
                return Err(format!("Failed to allocate OpenCL array of size {} on device {}", size, self.get_name()));
            }

            return Ok(self.array_registry.register_array(array, size));
        }
        return Err(format!("No OpenCL context for device {}", self.get_name()));
    }
    
    fn set_array(&self, id : u32, array : &SizedBuffer) -> Result<(),String> {
        if let Some(command_queue) = self.command_queue {
            let cl_mem = self.array_registry.get_array(id);
            if array.get_raw_size() != self.array_registry.get_array_size(id) {
                return Err(format!("OpenCL array {} and SizedBuffer differ in size ({} != {})", id, self.array_registry.get_array_size(id), array.get_raw_size()));
            }

            let retcode = unsafe { clEnqueueWriteBuffer(command_queue, cl_mem, true as u32, 0, array.get_raw_size(), array.get_raw_ptr(), 0, ptr::null(), ptr::null()) };
            if retcode != CL_SUCCESS {
                return Err(format!("Failed to read {} bytes in OpenCL array {}", array.get_raw_size(), id));
            }

            return Ok(());
        }
        return Err(format!("No OpenCL command queue for device {}", self.get_name()));
    }

    fn get_array(&self, id : u32, array : &mut SizedBufferMut) -> Result<(),String> {
        if let Some(command_queue) = self.command_queue {
            let cl_mem = self.array_registry.get_array(id);
            if array.get_raw_size() != self.array_registry.get_array_size(id) {
                return Err(format!("OpenCL array {} and SizedBuffer differ in size ({} != {})", id, self.array_registry.get_array_size(id), array.get_raw_size()));
            }

            let retcode = unsafe { clEnqueueReadBuffer(command_queue, cl_mem, true as u32, 0, array.get_raw_size(), array.get_raw_ptr_mut(), 0, ptr::null(), ptr::null()) };
            if retcode != CL_SUCCESS {
                return Err(format!("Failed to write {} bytes in OpenCL array {}", array.get_raw_size(), id));
            }

            return Ok(());
        }
        return Err(format!("No OpenCL command queue for device {}", self.get_name()));
    }

    fn delete_array(&mut self, id : u32) -> Result<(),String> {
        let array = self.array_registry.get_array(id);
        self.array_registry.unregister_array(id);
        let retcode = unsafe { clReleaseMemObject(array) };
        if retcode != CL_SUCCESS {
            return Err(format!("Failed to release OpenCL array {} on device {}", id, self.get_name()));
        }
        return Ok(());
    }

    fn finalize(&mut self) -> Result<(),String> {
        let mut retcode = CL_SUCCESS;

        if let Some(command_queue) = self.command_queue {
            retcode = unsafe { clReleaseCommandQueue(command_queue) };
        }
        self.command_queue = None;
        if retcode != CL_SUCCESS {
            return Err(format!("Failed to release OpenCL command queue for device {}", self.get_name()));
        }

        if let Some(context) = self.context {
            retcode = unsafe { clReleaseContext(context) };
        }
        self.context = None;
        if retcode != CL_SUCCESS {
            return Err(format!("Failed to release OpenCL context for device {}", self.get_name()));
        }

        return Ok(());
    }
}

impl OpenCL {
    pub fn get_platform_list() -> Result<Vec<CLPlatform>, i32> {
        let mut num_platforms = 0u32;
        let retcode = unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut num_platforms) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }

        let mut platform_ids : Vec<CLPlatformId> = repeat(ptr::null()).take(num_platforms as usize).collect();
        let retcode = unsafe { clGetPlatformIDs(num_platforms, platform_ids.as_mut_ptr(), ptr::null_mut()) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }

        let mut platforms = Vec::<CLPlatform>::with_capacity(num_platforms as usize);
        for platform_id in platform_ids {
            platforms.push(try!(CLPlatform::new(platform_id)));
        }
        
        return Ok(platforms);
    }

    pub fn get_device_list(platform : &CLPlatform) -> Result<Vec<CLDevice>, i32> {
        let mut num_devices = 0u32;
        let retcode = unsafe { clGetDeviceIDs(platform.get_id(), CL_DEVICE_TYPE_ALL, 0, ptr::null_mut(), &mut num_devices) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }

        let mut device_ids : Vec<CLDeviceId> = repeat(ptr::null()).take(num_devices as usize).collect();
        let retcode = unsafe { clGetDeviceIDs(platform.get_id(), CL_DEVICE_TYPE_ALL, num_devices, device_ids.as_mut_ptr(), ptr::null_mut()) };
        if retcode != CL_SUCCESS {
            return Err(retcode);
        }

        let mut devices = Vec::<CLDevice>::with_capacity(num_devices as usize);
        for device_id in device_ids {
            devices.push(try!(CLDevice::new(device_id)));
        }
        
        return Ok(devices);
    }
}
