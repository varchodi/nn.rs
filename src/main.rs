use nn::maths::Matrix;
use rand::Rng;

fn f_wb(x: &Matrix<f32>, w: &Matrix<f32>, b: f32) -> Matrix<f32> {
    // x * w.n_add(b)
    let x = x.n_mult(1.0); //conv &m to m
    let w = w.n_mult(1.0);
    (x * w).n_add(b)
}

fn mse(prediction: &Matrix<f32>, output: &Matrix<f32>) -> f32 {
    let err = prediction.clone() - output.clone();
    err.square().sum() / (prediction.dim.0 as f32) // Normalize by number of samples
}

fn cost(x: &Matrix<f32>, w: &Matrix<f32>, y: &Matrix<f32>, b: f32) -> f32 {
    mse(&f_wb(x, w, b), y)
}

/// Compute parameter gradient using limit formula **d/dx=(f(x+h)-f(x))/h (h -> -inf)**;   
/// ### Args :
/// - x (inputs)  : f32 Matrix  
/// - w (weights) : f32 Matrix
/// - y (output)  : f32 Matrix   
/// - f (modal fx): (x,w,b)->Matrix    
/// - b (biase)   : f32      
///
/// ### Output:    
/// (dw<Matrix>, db<Matrix>,cost<Matrix>)    
fn gradient(
    x: &Matrix<f32>,
    w: &Matrix<f32>,
    cost_f: fn(x: &Matrix<f32>, w: &Matrix<f32>, y: &Matrix<f32>, b: f32) -> f32,
    y: &Matrix<f32>,
    b: f32,
) -> (f32, f32) {
    let (m, n) = x.dim;
    let h: f32 = 1e-6;

    let dw = (cost_f(&x, &w.n_add(h), &y, b) - cost_f(&x, &w, &y, b)) / h;
    let db = (cost_f(&x, &w, &y, b + h) - cost_f(&x, &w, &y, b)) / h;

    // let db = Matrix::fill((1, 1), 0.1).sum();
    let dw = dw / m as f32;
    let db = db / m as f32;
    (dw, db)
}

fn gradient_descent(
    x: &Matrix<f32>,
    y: &Matrix<f32>,
    w_in: &Matrix<f32>,
    b_in: f32,
    cost_function: fn(x: &Matrix<f32>, w: &Matrix<f32>, y: &Matrix<f32>, b: f32) -> f32,
    gradient_function: fn(
        x: &Matrix<f32>,
        w: &Matrix<f32>,
        cost_f: fn(x: &Matrix<f32>, w: &Matrix<f32>, y: &Matrix<f32>, b: f32) -> f32,
        y: &Matrix<f32>,
        b: f32,
    ) -> (f32, f32),
    alpha: f32,
    iterations: usize,
) -> (Matrix<f32>, f32) {
    let mut w = w_in.n_add(0.0);
    let mut b = b_in;
    for i in 0..iterations {
        let (dw, db) = gradient_function(&x, &w, cost_function, &y, b);
        w = w.n_subs(alpha * dw);
        b = b - (alpha * db);
        let cost = cost_function(&x, &w, &y, b);
        println!("cost: {:?}", cost);
        println!(
            "Iteration: {:?} | w: :{:?} | b: {:?} | cost: ${:?} || dw: {:?} db:{:?}",
            i,
            w.clone().data,
            b,
            cost,
            dw,
            db
        );
    }

    (w, b)
}

fn main() {
    // let sum = add(1, 2);
    let mut rng = rand::thread_rng();
    let random_w: f32 = rng.gen();
    let random_b: f32 = rng.r#gen();

    let input: Matrix<f32> = Matrix {
        data: vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        dim: (4, 2),
    };

    let output: Matrix<f32> = Matrix {
        data: vec![0.0, 1.0, 0.0, 0.0],
        dim: (4, 1),
    };

    let weight_in: Matrix<f32> = Matrix {
        data: vec![random_w; 2],
        dim: (2, 1),
    };

    let (w, b) = gradient_descent(
        &input, &output, &weight_in, random_b, cost, gradient, 0.1, 5000,
    );
    // let mut w = weight_in;
    // for _ in 0..10 {
    //     let j = cost(&input, &w, &output, 1.00);
    //     let dj_dw = cost(&input, &w.n_add(1e-3), &output, 1.0);
    //     let costy = dj_dw;
    //     w = w.n_subs(costy * 0.01);
    //     println!("cost :{:?}", j);
    // }
    // // j.print();
    // // dj_dw.print();
    // w.print();

    let tr = f_wb(&input, &w, b);
    tr.print();
}
