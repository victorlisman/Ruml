use super::tensor_error::TensorError;

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        if data.len() != shape.iter().product() {
            return Err(TensorError::InvalidShape);
        }
        Ok(Tensor { data, shape })
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn multiply(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>
    where
        T: std::ops::Mul<Output = T> + Copy,
    {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a * *b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn subtract(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>
    where
        T: std::ops::Sub<Output = T> + Copy,
    {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn divide(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>
    where
        T: std::ops::Div<Output = T> + Copy,
    {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch);
        }
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a / *b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn dot(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>
    where
        T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default,
    {
        if self.shape[1] != other.shape[0] {
            return Err(TensorError::ShapeMismatch);
        }
        let mut data = Vec::with_capacity(self.shape[0] * other.shape[1]);
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                let mut sum = T::default();
                for k in 0..self.shape[1] {
                    sum = sum + self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
                }
                data.push(sum);
            }
        }
        Tensor::new(data, vec![self.shape[0], other.shape[1]])
    }

    pub fn transpose(&self) -> Tensor<T>
    where
        T: Copy,
    {
        let mut data = Vec::with_capacity(self.data.len());
        for i in 0..self.shape[1] {
            for j in 0..self.shape[0] {
                data.push(self.data[j * self.shape[1] + i]);
            }
        }
        Tensor::new(data, vec![self.shape[1], self.shape[0]]).unwrap()
    }

    pub fn pow(&self, power: u32) -> Tensor<T>
    where
        T: std::ops::Mul<Output = T> + Copy + Default,
    {
        let data: Vec<T> = self
            .data
            .iter()
            .map(|x| (0..power).fold(T::default(), |acc, _| acc * *x))
            .collect();
        Tensor::new(data, self.shape.clone()).unwrap()
    }

    pub fn multiply_scalar(&self, scalar: T) -> Tensor<T>
    where
        T: std::ops::Mul<Output = T> + Copy,
    {
        let data: Vec<T> = self.data.iter().map(|x| *x * scalar).collect();
        Tensor::new(data, self.shape.clone()).unwrap()
    }

    pub fn mean(&self) -> T
    where
        T: std::ops::Add<Output = T> + std::ops::Div<Output = T> + Copy + Default + From<f32>,
    {
        let sum: T = self.data.iter().copied().fold(T::default(), |acc, x| acc + x);
        sum / T::from(self.data.len() as f32)
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor<T>
    where
        T: Default + Copy,
    {
        let data = vec![T::default(); shape.iter().product()];
        Tensor::new(data, shape).unwrap()
    }

    pub fn ones(shape: Vec<usize>) -> Tensor<T>
    where
        T: Default + std::ops::Add<Output = T> + Copy,
    {
        let data = vec![(T::default() + T::default() + T::default()); shape.iter().product()];
        Tensor::new(data, shape).unwrap()
    }

    pub fn item(&self) -> T
    where
        T: Copy,
    {
        assert_eq!(self.data.len(), 1);
        self.data[0]
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor<T>, TensorError>
    where
        T: Copy,
    {
        if self.data.len() != shape.iter().product() {
            return Err(TensorError::InvalidShape);
        }
        Tensor::new(self.data.clone(), shape)
    }
}
