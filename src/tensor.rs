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

    /// Performs a dot product between two tensors along the specified axes.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to dot with.
    /// * `self_axis` - The axis of the first tensor to contract along.
    /// * `other_axis` - The axis of the second tensor to contract along.
    ///
    /// # Returns
    ///
    /// * `Ok(Tensor<T>)` - The result of the dot product.
    /// * `Err(TensorError::ShapeMismatch)` - If the dimensions are not aligned for dot product.
    pub fn dot(&self, other: &Tensor<T>, self_axis: usize, other_axis: usize) -> Result<Tensor<T>, TensorError>
    where
        T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default,
    {
        if self.shape.dims[self_axis] != other.shape.dims[other_axis] {
            return Err(TensorError::ShapeMismatch);
        }

        // Calculate the shape of the resulting tensor
        let mut new_shape = self.shape.dims.clone();
        new_shape.remove(self_axis);
        let mut other_shape = other.shape.dims.clone();
        other_shape.remove(other_axis);
        new_shape.extend(other_shape);

        let mut data = vec![T::default(); new_shape.iter().product()];

        let self_strides = self.compute_strides(&self.shape.dims);
        let other_strides = other.compute_strides(&other.shape.dims);
        let new_strides = self.compute_strides(&new_shape);

        for self_index_flat in 0..self.shape.size() {
            let self_index = self.index_from_flat(self_index_flat, &self_strides);
            let mut self_index_no_contract = self_index.clone();
            self_index_no_contract.remove(self_axis);

            for other_index_flat in 0..other.shape.size() {
                let other_index = other.index_from_flat(other_index_flat, &other_strides);
                if self_index[self_axis] == other_index[other_axis] {
                    let mut other_index_no_contract = other_index.clone();
                    other_index_no_contract.remove(other_axis);

                    let mut result_index = self_index_no_contract.clone();
                    result_index.extend(other_index_no_contract);

                    let result_index_flat = self.flat_from_index(&result_index, &new_strides);
                    data[result_index_flat] = data[result_index_flat] +
                        self.data[self_index_flat] * other.data[other_index_flat];
                }
            }
        }

        Tensor::new(data, new_shape)
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

    /// Transposes the tensor by permuting its axes.
    ///
    /// # Arguments
    ///
    /// * `perm` - A permutation of the tensor's axes.
    ///
    /// # Returns
    ///
    /// * `Result<Tensor<T>, TensorError>` - The transposed tensor or an error if the permutation is invalid.
    pub fn transpose(&self, perm: Vec<usize>) -> Result<Tensor<T>, TensorError> {
        if perm.len() != self.shape.ndims() {
            return Err(TensorError::InvalidShape);
        }

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape.dims[i]).collect();
        let mut data = vec![T::default(); self.data.len()];

        let old_strides = self.compute_strides(&self.shape.dims);
        let new_strides = self.compute_strides(&new_shape);

        for (i, &value) in self.data.iter().enumerate() {
            let old_indices = self.index_from_flat(i, &old_strides);
            let new_indices: Vec<usize> = perm.iter().map(|&j| old_indices[j]).collect();
            let new_idx = self.flat_from_index(&new_indices, &new_strides);

            data[new_idx] = value;
        }

        Tensor::new(data, new_shape)
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

    /// Helper function to compute the strides of a tensor given its shape.
    fn compute_strides(&self, shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Helper function to convert a flat index to a multi-dimensional index.
    fn index_from_flat(&self, flat_index: usize, strides: &[usize]) -> Vec<usize> {
        let mut index = vec![0; strides.len()];
        let mut remaining = flat_index;
        for i in 0..strides.len() {
            index[i] = remaining / strides[i];
            remaining %= strides[i];
        }
        index
    }

    /// Helper function to convert a multi-dimensional index to a flat index.
    fn flat_from_index(&self, index: &[usize], strides: &[usize]) -> usize {
        index.iter().zip(strides).map(|(&i, &s)| i * s).sum()
    }

}
