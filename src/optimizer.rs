use crate::tensor::Tensor;
use crate::model::Model;
use std::ops::{Add, Mul, Div, Sub};

trait Optimizer<T> {
    fn step(&mut self, model: &mut dyn Model<T>);
}

pub struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl<T> Optimizer<T> for SGD {
    fn step(&mut self, model: &mut dyn Model<T>) {
        unimplemented!()
    }
}

