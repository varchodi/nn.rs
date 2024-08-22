use std::ops::Mul;

use crate::maths::Matrix;

pub fn sigmoid(z: &Matrix<f32>) -> Matrix<f32> {
    let data: Vec<f32> = z
        .clone()
        .data
        .iter_mut()
        .map(|x| 1.0 / (1.0 + (x.mul(-1.0).exp())))
        .collect();

    let out: Matrix<f32> = Matrix { data, dim: z.dim };
    out
}

#[cfg(test)]
mod cool {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let mtx: Matrix<f32> = Matrix {
            data: vec![0.2, 0.7, 0.5, 0.0],
            dim: (1, 4),
        };

        let mtx = sigmoid(&mtx);
        // let output = sigmoid(&z);
        let predictions: Vec<u8> = mtx
            .data
            .iter()
            .map(|x| {
                if *x as f32 > 0.59999999999999999 as f32 {
                    1
                } else {
                    0
                }
            })
            .collect();
        assert_eq!(
            predictions,
            vec![0, 1, 1, 0],
            "Sigmoid function behave anormally"
        );
    }
}
