pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Tensor { data, shape }
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn multiply(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::ops::Mul<Output = T> + Copy,
    {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a * *b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn add(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn subtract(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::ops::Sub<Output = T> + Copy,
    {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn divide(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::ops::Div<Output = T> + Copy,
    {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a / *b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn dot(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy,
    {
        assert_eq!(self.shape[1], other.shape[0]);
        let mut data = Vec::new();
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                let mut sum = self.data[i * self.shape[1]] * other.data[j];
                for k in 1..self.shape[1] {
                    sum = sum + self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
                }
                data.push(sum);
            }
        }
        Tensor::new(data, vec![self.shape[0], other.shape[1]])
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor<T>
    where
        T: Default + Copy,
    {
        let data = vec![T::default(); shape.iter().product()];
        Tensor::new(data, shape)
    }

    pub fn ones(shape: Vec<usize>) -> Tensor<T>
    where
        T: Default + std::ops::Add<Output = T> + Copy,
    {
        let data = vec![T::default() + T::default(); shape.iter().product()];
        Tensor::new(data, shape)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.shape(), &vec![2, 3]);
    }

    #[test]
    fn test_multiply() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1);

        let data2 = vec![7, 8, 9, 10, 11, 12];
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2);

        let result = tensor1.multiply(&tensor2);
        assert_eq!(result.data, vec![7, 16, 27, 40, 55, 72]);
    }

    #[test]
    fn test_add() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1);

        let data2 = vec![7, 8, 9, 10, 11, 12];
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2);

        let result = tensor1.add(&tensor2);
        assert_eq!(result.data, vec![8, 10, 12, 14, 16, 18]);
    }

    #[test]
    fn test_subtract() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1);

        let data2 = vec![7, 8, 9, 10, 11, 12];
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2);

        let result = tensor1.subtract(&tensor2);
        assert_eq!(result.data, vec![-6, -6, -6, -6, -6, -6]);
    }

    #[test]
    fn test_divide() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1);

        let data2 = vec![7, 8, 9, 10, 11, 12];
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2);

        let result = tensor1.divide(&tensor2);
        assert_eq!(result.data, vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_dot() {
        let data1 = vec![1, 2, 3, 4];
        let shape1 = vec![2, 2];
        let tensor1 = Tensor::new(data1, shape1);

        let data2 = vec![5, 6, 7, 8];
        let shape2 = vec![2, 2];
        let tensor2 = Tensor::new(data2, shape2);

        let result = tensor1.dot(&tensor2);
        assert_eq!(result.data, vec![19, 22, 43, 50]);
    }

}