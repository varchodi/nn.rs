use crate::maths::Matrix;

pub fn sigmoid(z: &Matrix<f32>) -> Matrix<f32> {
    let data: Vec<f32> = z
        .clone()
        .data
        .iter_mut()
        .map(|x| 1.0 / (1.0 + (x.exp())))
        .collect();
    let out: Matrix<f32> = Matrix { data, dim: z.dim };
    out
}
