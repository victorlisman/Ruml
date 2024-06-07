use crate::tensor::Tensor;

pub trait Model<T> {
    fn forward(&self, input: &Tensor<T>) -> Tensor<T>;
    fn as_any(&mut self) -> &mut dyn std::any::Any;
}