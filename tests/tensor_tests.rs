extern crate ruml;
use ruml::tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;
    use ruml::shape::Shape; 

    #[test]
    fn test_tensor() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data, shape).unwrap();
        assert_eq!(tensor.shape(), &Shape::new(vec![2, 3]));
    }

    #[test]
    fn test_multiply() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1).unwrap();

        let data2 = vec![7, 8, 9, 10, 11, 12];
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2).unwrap();

        let result = tensor1.multiply(&tensor2).unwrap();
        assert_eq!(*result.data(), vec![7, 16, 27, 40, 55, 72]);
    }

    #[test]
    fn test_add() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1).unwrap();

        let data2 = vec![7, 8, 9, 10, 11, 12];
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2).unwrap();

        let result = tensor1.add(&tensor2).unwrap();
        assert_eq!(*result.data(), vec![8, 10, 12, 14, 16, 18]);
    }

    #[test]
    fn test_subtract() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1).unwrap();

        let data2 = vec![7, 8, 9, 10, 11, 12];
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2).unwrap();

        let result = tensor1.subtract(&tensor2).unwrap();
        assert_eq!(*result.data(), vec![-6, -6, -6, -6, -6, -6]);
    }

    #[test]
    fn test_divide() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1).unwrap();

        let data2 = vec![7, 8, 9, 10, 11, 12];
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2).unwrap();

        let result = tensor1.divide(&tensor2).unwrap();
        assert_eq!(*result.data(), vec![1 / 7, 2 / 8, 3 / 9, 4 / 10, 5 / 11, 6 / 12]);
    }

    #[test]
    fn test_dot() {
        let data1 = vec![1, 2, 3, 4];
        let shape1 = vec![2, 2];
        let tensor1 = Tensor::new(data1, shape1).unwrap();

        let data2 = vec![5, 6, 7, 8];
        let shape2 = vec![2, 2];
        let tensor2 = Tensor::new(data2, shape2).unwrap();

        let result = tensor1.dot(&tensor2, 1, 0).unwrap();
        assert_eq!(*result.data(), vec![19, 22, 43, 50]);
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, shape).unwrap();

        let result = tensor.mean();
        assert_eq!(result, 2.5);
    }

    #[test]
    fn test_reshape() {
        let data = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, shape).unwrap();

        let new_shape = vec![4];
        let reshaped = tensor.reshape(new_shape).unwrap();
        assert_eq!(reshaped.shape(), &Shape::new(vec![4]));
        assert_eq!(reshaped.data(), &vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_transpose() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data, shape).unwrap();

        let transposed = tensor.transpose(vec![1, 0]).unwrap();
        assert_eq!(transposed.shape(), &Shape::new(vec![3, 2]));
        assert_eq!(transposed.data(), &vec![1, 4, 2, 5, 3, 6]);
    }
}
