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

pub fn tanh(z: &Matrix<f32>) -> Matrix<f32> {
    // let data=z.data.iter_mut().map(|x| )
    (sigmoid(&z).n_mult(2 as f32)).n_subs(1 as f32)
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
    }
}
