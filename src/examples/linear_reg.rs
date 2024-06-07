extern crate ruml;
use ruml::tensor::tensor::Tensor;

pub struct LinearRegression {
    lr: f64,
    epochs: usize,
    w: Option<Tensor<f64>>,
    b: f64,
}

impl LinearRegression {
    pub fn new(lr: f64, epochs: usize) -> Self {
        Self {
            lr,
            epochs,
            w: None,
            b: 0.0,
        }
    }

    pub fn fit(&mut self, X: &Tensor<f64>, y: &Tensor<f64>) {
        let num_samples = X.shape().dims[0];
        let num_features = X.shape().dims[1];

        self.w = Some(Tensor::new(vec![0.0; num_features], vec![num_features, 1]).unwrap());

        for epoch in 0..self.epochs {
            let y_pred = X.dot(self.w.as_ref().unwrap()).unwrap().add_scalar(self.b);

            let dw = X.transpose().dot(&y_pred.subtract(y).unwrap()).unwrap().multiply_scalar(1.0 / num_samples as f64);
            let db = y_pred.subtract(y).unwrap().sum() * (1.0 / num_samples as f64);

            self.w = Some(self.w.as_ref().unwrap().subtract(&dw.multiply_scalar(self.lr)).unwrap());
            self.b -= db * self.lr;


            println!("Epoch {}: w = {:?}, b = {}, dw = {:?}, db = {}", epoch, self.w.as_ref().unwrap().data(), self.b, dw.data(), db);

            if self.w.as_ref().unwrap().data().iter().any(|&val| val.abs() > 1e6) || self.b.abs() > 1e6 {
                println!("Overflow detected. Stopping training.");
                break;
            }
        }
    }

    pub fn predict(&self, X: &Tensor<f64>) -> Tensor<f64> {
        X.dot(self.w.as_ref().unwrap()).unwrap().add_scalar(self.b)
    }

    pub fn mse(&self, y_true: &Tensor<f64>, y_pred: &Tensor<f64>) -> f64 {
        y_true.subtract(y_pred).unwrap().data().iter().map(|&x| x * x).sum::<f64>() / y_true.shape().dims[0] as f64
    }

    pub fn r2_score(&self, y_true: &Tensor<f64>, y_pred: &Tensor<f64>) -> f64 {
        let numerator = y_true.subtract(y_pred).unwrap().data().iter().map(|&x| x * x).sum::<f64>();
        let mean_y_true = y_true.mean();
        let denominator = y_true.data().iter().map(|&x| (x - mean_y_true).powi(2)).sum::<f64>();
        1.0 - (numerator / denominator)
    }

    pub fn evaluate(&self, X: &Tensor<f64>, y: &Tensor<f64>) -> (f64, f64) {
        let y_pred = self.predict(X);
        (self.mse(y, &y_pred), self.r2_score(y, &y_pred))
    }
}

fn main() {
    let data_x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let data_y = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    
    let mean_x = data_x.iter().sum::<f64>() / data_x.len() as f64;
    let std_x = (data_x.iter().map(|&x| (x - mean_x).powi(2)).sum::<f64>() / data_x.len() as f64).sqrt();
    let normalized_x: Vec<f64> = data_x.iter().map(|&x| (x - mean_x) / std_x).collect();

    let shape_x = vec![10, 1];
    let shape_y = vec![10, 1];
    
    let X = Tensor::new(normalized_x, shape_x).unwrap();
    let y = Tensor::new(data_y, shape_y).unwrap();
    
    let mut model = LinearRegression::new(0.05, 100);
    model.fit(&X, &y);
    
    let predictions = model.predict(&X);
    let (mse, r2) = model.evaluate(&X, &y);
    
    println!("Predictions: {:?}", predictions.data());
    println!("Mean Squared Error: {}", mse);
    println!("R2 Score: {}", r2);
}