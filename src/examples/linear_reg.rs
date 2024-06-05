use tensor::tensor::Tensor;


fn main() {
    let data1 = vec![1, 2, 3, 4, 5, 6];
    let shape1 = vec![2, 3];
    let tensor1 = Tensor::new(data1, shape1);

    let data2 = vec![7, 8, 9, 10, 11, 12];
    let shape2 = vec![2, 3];
    let tensor2 = Tensor::new(data2, shape2);

    let result = tensor1.add(&tensor2);
    println!("{:?}", result.data);
}