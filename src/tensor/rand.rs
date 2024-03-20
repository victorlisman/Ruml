struct Rand {
    tensor: Tensor,
    shape: Vec<i32>,
    seed: i32,
}

impl Rand {
    pub fn new(shape: Vec<i32>, seed: i32) -> Rand {
        let tensor = Tensor::new(vec![], shape.clone());
        Rand {
            tensor,
            shape,
            seed,
        }
    }

    pub fn fill(&mut self) {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..self.tensor.data().len())
            .map(|_| rng.gen_range(0.0, 1.0))
            .collect();
        self.tensor = Tensor::new(data, self.shape.clone());
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
}

