use crate::tensor_error::TensorError;
use super::shape::Shape;
use std::ops::{Add, Sub, Mul, Div};

/// A generic Tensor struct that holds multi-dimensional array data and its shape.
#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T> Tensor<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    /// Creates a new tensor with the given data and shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector containing the elements of the tensor.
    /// * `shape` - A vector specifying the dimensions of the tensor.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor<T>)` - If the data length matches the shape's size.
    /// * `Err(TensorError::InvalidShape)` - If the data length does not match the shape's size.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let shape = Shape::new(shape);
        if data.len() != shape.size() {
            return Err(TensorError::InvalidShape);
        }
        Ok(Tensor { data, shape })
    }

    /// Performs a dot product between two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to dot with.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor<T>)` - The result of the dot product.
    /// * `Err(TensorError::InvalidShape)` - If either tensor is not 2-dimensional.
    /// * `Err(TensorError::ShapeMismatch)` - If the dimensions are not aligned for dot product.
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

    /// Computes the sum of all elements in the tensor.
    ///
    /// # Returns
    ///
    /// * `T` - The sum of all elements.
    pub fn sum(&self) -> T {
        self.data.iter().copied().fold(T::default(), |acc, x| acc + x)
    }

    /// Computes the mean of all elements in the tensor.
    ///
    /// # Returns
    ///
    /// * `T` - The mean of all elements.
    pub fn mean(&self) -> T
    where
        T: From<f32>,
    {
        self.sum() / T::from(self.data.len() as f32)
    }

    /// Transposes the tensor.
    ///
    /// # Returns
    ///
    /// * `Tensor<T>` - The transposed tensor.
    pub fn transpose(&self) -> Tensor<T> {
        let mut data = Vec::with_capacity(self.data.len());
        for i in 0..self.shape.dims[1] {
            for j in 0..self.shape.dims[0] {
                data.push(self.data[j * self.shape.dims[1] + i]);
            }
        }
        Tensor::new(data, vec![self.shape.dims[1], self.shape.dims[0]]).unwrap()
    }

    /// Adds a scalar to each element of the tensor.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar to add.
    ///
    /// # Returns
    ///
    /// * `Tensor<T>` - The result tensor.
    pub fn add_scalar(&self, scalar: T) -> Tensor<T>
    where
        T: Copy + Add<Output = T>,
    {
        let data: Vec<T> = self.data.iter().map(|&x| x + scalar).collect();
        Tensor::new(data, self.shape.dims.clone()).unwrap()
    }

    /// Subtracts another tensor from this tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to subtract.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor<T>)` - The result of the subtraction.
    /// * `Err(TensorError::ShapeMismatch)` - If the shapes do not match.
    pub fn subtract(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape.dims != other.shape.dims {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect();
        Tensor::new(data, self.shape.dims.clone())
    }

    /// Multiplies each element of the tensor by a scalar.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar to multiply.
    ///
    /// # Returns
    ///
    /// * `Tensor<T>` - The result tensor.
    pub fn multiply_scalar(&self, scalar: T) -> Tensor<T>
    where
        T: Copy + Mul<Output = T>,
    {
        let data: Vec<T> = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::new(data, self.shape.dims.clone()).unwrap()
    }

    /// Multiplies this tensor with another tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to multiply.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor<T>)` - The result of the multiplication.
    /// * `Err(TensorError::ShapeMismatch)` - If the shapes do not match.
    pub fn multiply(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape.dims != other.shape.dims {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self.data.iter().zip(&other.data).map(|(&a, &b)| a * b).collect();
        Tensor::new(data, self.shape.dims.clone())
    }

    /// Adds another tensor to this tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to add.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor<T>)` - The result of the addition.
    /// * `Err(TensorError::ShapeMismatch)` - If the shapes do not match.
    pub fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape.dims != other.shape.dims {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect();
        Tensor::new(data, self.shape.dims.clone())
    }

    /// Divides this tensor by another tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to divide by.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor<T>)` - The result of the division.
    /// * `Err(TensorError::ShapeMismatch)` - If the shapes do not match.
    pub fn divide(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape.dims != other.shape.dims {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self.data.iter().zip(&other.data).map(|(&a, &b)| a / b).collect();
        Tensor::new(data, self.shape.dims.clone())
    }

    /// Reshapes the tensor to the new shape.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The new shape for the tensor.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor<T>)` - The reshaped tensor.
    /// * `Err(TensorError::InvalidShape)` - If the total number of elements does not match.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor<T>, TensorError> {
        if self.shape.size() != new_shape.iter().product() {
            return Err(TensorError::InvalidShape);
        }
        Tensor::new(self.data.clone(), new_shape)
    }

    /// Returns a reference to the shape of the tensor.
    ///
    /// # Returns
    ///
    /// * `&Shape` - The shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns a reference to the data of the tensor.
    ///
    /// # Returns
    ///
    /// * `&Vec<T>` - The data of the tensor.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }
}
