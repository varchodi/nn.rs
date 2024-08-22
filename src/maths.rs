use std::{
    f32,
    ops::{Add, Mul, Sub},
};
pub enum OtherMatrix<T> {
    A(T),
    B(Matrix<T>),
}

pub enum Term {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    ///dim(nrows,ncols)
    pub dim: (usize, usize),
}

impl Matrix<f32> {
    pub fn new(nrow: usize, ncol: usize) -> Matrix<f32> {
        Matrix {
            data: vec![0.00; nrow * ncol],
            dim: (nrow, ncol),
        }
    }

    /// Create a Matrix dim filled with x element   
    /// Args:   
    /// - dim     :(nrow,ncols)    
    /// - element : f32    
    ///
    ///Return  : Matrix<f32>    
    pub fn fill(dim: (usize, usize), element: f32) -> Matrix<f32> {
        Matrix {
            data: vec![element; dim.0 * dim.1],
            dim,
        }
    }

    pub fn size(&self) -> usize {
        self.dim.0 * self.dim.1
    }

    // !! todo (by row ,or cls)
    pub fn sum(&self) -> f32 {
        let sum = self.data.iter().sum();
        sum
    }

    pub fn square(&self) -> Matrix<f32> {
        let mut output = self.clone();
        output.data = output.data.iter().map(|&x| x * x).collect();
        // let cool: Vec<f32> = output.data.iter().map(|&x| x * x).collect();
        output
    }

    pub fn alloc(&mut self, data: Vec<f32>) {
        // ?? fixable
        if self.size() < data.len() {
            panic!("unmatched Matrix sizes {:?} - {}", self.size(), data.len());
        };
        self.data = data;
    }

    pub fn get(&self, i: usize, j: usize) -> f32 {
        // i*cols+j
        self.data[i * self.dim.1 + j]
    }

    pub fn get_row(&self, nthrow: usize) -> Matrix<f32> {
        let mut out = Matrix::new(1, self.dim.1);
        for j in 0..self.dim.1 {
            out.data[j] = self.get(nthrow, j);
        }
        out
    }

    /// operations with integers (numbers)
    /// addition
    pub fn n_add(&self, num: f32) -> Matrix<f32> {
        let (rows, cols) = self.dim;
        let mut output = Matrix::new(rows, cols);
        for i in 0..self.data.len() {
            output.data[i] = self.data[i] + num;
        }

        output
    }
    // substraction
    pub fn n_subs(&self, num: f32) -> Matrix<f32> {
        let (rows, cols) = self.dim;
        let mut output = Matrix::new(rows, cols);
        for i in 0..self.data.len() {
            output.data[i] = self.data[i] - num;
        }

        output
    }
    // div
    pub fn n_div(&self, num: f32) -> Matrix<f32> {
        let (rows, cols) = self.dim;
        let mut output = Matrix::new(rows, cols);
        for i in 0..self.data.len() {
            output.data[i] = self.data[i] / num;
        }

        output
    }

    // multiplication
    pub fn n_mult(&self, num: f32) -> Matrix<f32> {
        let (rows, cols) = self.dim;
        let mut output = Matrix::new(rows, cols);
        for i in 0..self.data.len() {
            output.data[i] = self.data[i] * num;
        }

        output
    }

    // lineage
    // dot_product
    pub fn dot(&self, other: &Matrix<f32>) -> f32 {
        // let (rows, cols) = self.dim;
        if self.dim != other.dim {
            panic!(
                "Coud not peform operation dot, Unmached sizes {:?}@{:?}",
                self.dim, other.dim
            );
        }
        let mut solution = 0.0;
        for i in 0..self.data.len() {
            solution += self.data[i] * other.data[i];
        }
        solution
    }

    // print matrix data
    pub fn print(&self) {
        let (nrow, ncol) = self.dim;

        println!("[");
        for i in 0..nrow {
            print!("  [");
            for j in 0..ncol {
                print!(" {} ", self.data[i * ncol + j]);
            }
            println!("]");
        }
        println!("]");
    }
}

// Matrix Operations
// addition
impl Add for Matrix<f32> {
    type Output = Matrix<f32>;
    fn add(self, other: Matrix<f32>) -> Self::Output {
        let (rows, cols) = self.dim;

        if self.dim != other.dim {
            panic!(
                "Cannot peform Operation +: Unmatched matrix size: {:?} + {:?}",
                self.dim, other.dim
            );
        }
        // init output
        let mut output: Matrix<f32> = Matrix::new(rows, cols);
        // peform operation
        for i in 0..rows {
            for j in 0..cols {
                output.data[i * cols + j] = self.get(i, j) + other.get(i, j);
            }
        }

        output
    }
}

// Subraction
impl Sub for Matrix<f32> {
    type Output = Matrix<f32>;
    fn sub(self, other: Matrix<f32>) -> Self::Output {
        let (self_row, self_col) = self.dim;

        if self.dim != other.dim {
            panic!(
                "Cannot peform Operation -: Unmatched matrix size: {:?} - {:?}",
                self.dim, other.dim
            );
        }

        let mut output = Matrix::new(self_row, self_col);

        for i in 0..self_row {
            for j in 0..self_col {
                output.data[i * self_col + j] = self.get(i, j) - other.get(i, j);
            }
        }
        output
    }
}

// multiplication
impl Mul for Matrix<f32> {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        let (a_rows, a_cols) = self.dim;
        let (b_rows, b_cols) = other.dim;

        if a_cols != b_rows {
            panic!(
                "Error: cannot peform operations x, incopatible matries sizes {:?}x{:?}",
                self.dim, other.dim
            );
        }
        // resulting matrix same rows as a, and colums a b
        let mut result = Matrix::new(a_rows, b_cols);

        for i in 0..a_rows {
            for j in 0..b_cols {
                for k in 0..a_cols {
                    result.data[i * b_cols + j] += self.get(i, k) * other.get(k, j);
                }
            }
        }

        result
    }
}

//derivatives (parial with limits);
// pub fn p_derivative(m: &Matrix<f32>, term: Term) -> Matrix<f32> {
//     let m = m.n_add(0.0);

// }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_display() {
        let mat: Matrix<f32> = Matrix {
            dim: (1, 2),
            data: vec![1.0, 1.0],
        };
        mat.print();
    }
    #[test]
    fn test_addition() {
        let mut mat1 = Matrix::new(1, 2);
        mat1.alloc(vec![1.0, 2.0]);
        let mut mat2 = Matrix::new(1, 2);
        mat2.alloc(vec![3.0, 3.0]);
        let dim1 = mat1.dim;
        let dim2 = mat2.dim;

        let out = mat1 + mat2;
        println!("{:?}", out.data);
        assert_eq!(dim1, dim2, "Sizes match");
        assert_eq!(out.data, vec![4.0, 5.0], "Operations works");
    }
    #[test]
    fn test_substaction() {
        let mut mat1 = Matrix::new(1, 3);
        mat1.alloc(vec![1.0, 2.0, 5.0]);
        let mut mat2 = Matrix::new(1, 3);
        mat2.alloc(vec![3.0, 4.0, 7.0]);
        let dim1 = mat1.dim;
        let dim2 = mat2.dim;

        let out = mat1 - mat2;
        println!("{:?}", out.data);
        assert_eq!(dim1, dim2, "Sizes match");
        assert_eq!(out.data, vec![-2.0, -2.0, -2.0], "Operations works");
    }
    #[test]
    fn test_operations() {
        let mut mat = Matrix::new(1, 3);
        mat.alloc(vec![1.0, 2.0, 5.0]);
        assert_eq!(
            mat.n_add(2.0).data,
            vec![3.0, 4.0, 7.0],
            "Addition with n works"
        );
        assert_eq!(
            mat.n_subs(2.0).data,
            vec![-1.0, 0.0, 3.0],
            "Substraction with n works"
        );
        assert_eq!(
            mat.n_mult(2.0).data,
            vec![2.0, 4.0, 10.0],
            "Multiplication with n works"
        );
        assert_eq!(
            mat.n_div(2.0).data,
            vec![0.5, 1.0, 2.5],
            "Division with n works"
        );
    }
    #[test]
    fn test_dot_m_product() {
        let mut mat1 = Matrix::new(1, 3);
        mat1.alloc(vec![1.0, 2.0, 5.0]);
        let mut mat2 = Matrix::new(1, 3);
        mat2.alloc(vec![3.0, 4.0, 7.0]);

        let solution = mat1.dot(&mat2);
        assert_eq!(solution, 46.0, "Dot product works");
    }
    #[test]
    fn test_mult() {
        let mut mat1 = Matrix::new(3, 2);
        mat1.alloc(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut mat2 = Matrix::new(2, 3);
        mat2.alloc(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let solution = mat1 * mat2;
        assert_eq!(
            solution.data,
            vec![15.0, 18.0, 21.0, 33.0, 40.0, 47.0, 51.0, 62.0, 73.0],
            "Multiplication works"
        );
    }

    #[test]
    fn test_get_row() {
        let mat: Matrix<f32> = Matrix {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dim: (3, 2),
        };

        let sclic: Matrix<f32> = mat.get_row(2);
        sclic.print();
        assert_eq!(sclic.data, vec![5.0, 6.0], "Arrow scice works");
    }
}
