use crate::tensor_error::TensorError;
use super::shape::Shape;
use std::ops::{Add, Sub, Mul, Div};

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T> Tensor<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let shape = Shape::new(shape);
        if data.len() != shape.size() {
            return Err(TensorError::InvalidShape);
        }
        Ok(Tensor { data, shape })
    }

    pub fn dot(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>
    where
        T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default,
    {
        if self.shape.dims.len() != 2 || other.shape.dims.len() != 2 {
            return Err(TensorError::InvalidShape);
        }
        if self.shape.dims[1] != other.shape.dims[0] {
            return Err(TensorError::ShapeMismatch);
        }

        let rows = self.shape.dims[0];
        let cols = other.shape.dims[1];
        let common_dim = self.shape.dims[1];

        let mut data = vec![T::default(); rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                let mut sum = T::default();
                for k in 0..common_dim {
                    sum = sum + self.data[i * common_dim + k] * other.data[k * cols + j];
                }
                data[i * cols + j] = sum;
            }
        }

        Tensor::new(data, vec![rows, cols])
    }

    pub fn sum(&self) -> T {
        self.data.iter().copied().fold(T::default(), |acc, x| acc + x)
    }

    pub fn mean(&self) -> T
    where
        T: From<f32>,
    {
        self.sum() / T::from(self.data.len() as f32)
    }

    pub fn transpose(&self) -> Tensor<T> {
        let mut data = Vec::with_capacity(self.data.len());
        for i in 0..self.shape.dims[1] {
            for j in 0..self.shape.dims[0] {
                data.push(self.data[j * self.shape.dims[1] + i]);
            }
        }
        Tensor::new(data, vec![self.shape.dims[1], self.shape.dims[0]]).unwrap()
    }

    pub fn add_scalar(&self, scalar: T) -> Tensor<T>
    where
        T: Copy + Add<Output = T>,
    {
        let data: Vec<T> = self.data.iter().map(|&x| x + scalar).collect();
        Tensor::new(data, self.shape.dims.clone()).unwrap()
    }

    pub fn subtract(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape.dims != other.shape.dims {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect();
        Tensor::new(data, self.shape.dims.clone())
    }

    pub fn multiply_scalar(&self, scalar: T) -> Tensor<T>
    where
        T: Copy + Mul<Output = T>,
    {
        let data: Vec<T> = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::new(data, self.shape.dims.clone()).unwrap()
    }

    pub fn multiply(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape.dims != other.shape.dims {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self.data.iter().zip(&other.data).map(|(&a, &b)| a * b).collect();
        Tensor::new(data, self.shape.dims.clone())
    }

    pub fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape.dims != other.shape.dims {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect();
        Tensor::new(data, self.shape.dims.clone())
    }

    pub fn divide(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape.dims != other.shape.dims {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self.data.iter().zip(&other.data).map(|(&a, &b)| a / b).collect();
        Tensor::new(data, self.shape.dims.clone())
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor<T>, TensorError> {
        if self.shape.size() != new_shape.iter().product() {
            return Err(TensorError::InvalidShape);
        }
        Tensor::new(self.data.clone(), new_shape)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }
}