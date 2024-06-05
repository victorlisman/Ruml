use super::tensor::Tensor;
use rand::Rng;

struct Rand<T> {
    tensor: Tensor<T>,
    shape: Vec<usize>,
    seed: i32,
}

impl<T> Rand<T> {
    pub fn new(shape: Vec<usize>, seed: i32) -> Rand<T> {
        let tensor = Tensor::new(vec![], shape.clone()).unwrap();
        Rand {
            tensor,
            shape,
            seed,
        }
    }

    pub fn fill(&mut self) {
        unimplemented!();
    }

    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }
}

