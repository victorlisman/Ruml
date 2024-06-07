#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    pub fn ndims(&self) -> usize {
        self.dims.len()
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn dim(&self, index: usize) -> Option<usize> {
        self.dims.get(index).cloned()
    }
}
