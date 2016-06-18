pub trait NDIndex {

    fn inc_ro(&mut self, shape : &[usize]);

    fn dec_ro(&mut self, shape : &[usize]);

    fn inc_co(&mut self, shape : &[usize]);

    fn dec_co(&mut self, shape : &[usize]);

    fn is_zero(&mut self) -> bool;
    
    fn to_pos(&self, shape : &[usize], strides : &[usize]) -> usize;
}

impl NDIndex for [usize] {

    fn inc_ro(&mut self, shape : &[usize]) {
        let mut i = self.len();
        while i > 0 {
            self[i-1] += 1;
            if self[i-1] >= shape[i-1] {
                self[i-1] = 0;
                i -= 1;
            }
            else {
                break;
            }
        }
    }

    fn dec_ro(&mut self, shape : &[usize]) {
        let mut i = self.len();
        while i > 0 {
            if self[i-1] == 0 {
                self[i-1] = shape[i-1] - 1;
                i -= 1;
            }
            else {
                self[i-1] -= 1;
                break;
            }
        }
    }

    fn inc_co(&mut self, shape : &[usize]) {
        let mut i = 0;
        while i < self.len() {
            self[i] += 1;
            if self[i] >= shape[i] {
                self[i] = 0;
                i += 1;
            }
            else {
                break;
            }
        }
    }

    fn dec_co(&mut self, shape : &[usize]) {
        let mut i = 0;
        while i < self.len() {
            if self[i] ==  0 {
                self[i] = shape[i] - 1;
                i += 1;
            }
            else {
                self[i] -= 1;
                break;
            }
        }
    }

    fn is_zero(&mut self) -> bool {
        for i in 0..self.len() {
            if self[i] != 0 {
                return false;
            }
        }
        return true;
    }

    fn to_pos(&self, shape : &[usize], strides : &[usize]) -> usize {
        let mut pos = 0usize;
        for i in 0..self.len() {
            if self[i] >= shape[i] {
                panic!("NDIndex::to_pos(): idx is out of bound for dimension {} ({} >= {})", i, self[i], shape[i]);
            }
            pos += self[i] * strides[i];
        }
        return pos;
    }
}
