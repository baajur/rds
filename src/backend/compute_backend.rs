pub trait SizedBuffer {}

pub trait ComputeBackend {

    fn init();

    fn createCBArray() -> u32;
    
    fn setCBArray(id : u32, array : &SizedBuffer);

    fn getCBArray(id : u32, array : &mut SizedBuffer);

    fn deleteCBArray(id : u32);

    fn finalize();
}
