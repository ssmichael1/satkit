/// Unscented Kalman Filter
///
/// Uses nalgebra
///
///
//

type Vector<const T: usize> = nalgebra::SVector<f64, T>;
type Matrix<const M: usize, const N: usize> = nalgebra::SMatrix<f64, M, N>;
use crate::utils::{skerror, SKResult};

pub struct UKF<const N: usize> {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
    pub weight_m: Vec<f64>,
    pub weight_c: Vec<f64>,
    pub x: Vector<N>,
    pub p: Matrix<N, N>,
}

impl<const N: usize> UKF<N> {
    fn weight_m(alpha: f64, kappa: f64) -> Vec<f64> {
        let mut weight_m = Vec::<f64>::with_capacity(2 * N + 1);
        let den: f64 = alpha.powi(2) as f64 * (N as f64 + kappa);
        weight_m.push(1.0 - N as f64 / den);
        for _i in 1..2 * N + 1 {
            weight_m.push(1.0 / (2.0 * den));
        }
        weight_m
    }

    fn weight_c(alpha: f64, beta: f64, kappa: f64) -> Vec<f64> {
        let mut weight_c = Vec::<f64>::with_capacity(2 * N + 1);
        let den: f64 = alpha.powi(2) as f64 * (N as f64 + kappa);
        weight_c.push(2.0 - alpha.powi(2) as f64 + beta - N as f64 / den);
        for _i in 1..2 * N + 1 {
            weight_c.push(1.0 / (2.0 * den));
        }
        weight_c
    }

    /// Constructor with default values
    pub fn new_default() -> Self {
        Self {
            alpha: 0.001,
            beta: 2.0,
            kappa: 0.0,
            weight_m: Self::weight_m(0.001, 0.0),
            weight_c: Self::weight_c(0.001, 2.0, 0.0),
            x: Vector::<N>::zeros(),
            p: Matrix::<N, N>::zeros(),
        }
    }

    /// Constructor with custom values
    pub fn new(alpha: f64, beta: f64, kappa: f64) -> Self {
        Self {
            alpha,
            beta,
            kappa,
            weight_m: Self::weight_m(alpha, kappa),
            weight_c: Self::weight_c(alpha, beta, kappa),
            x: Vector::<N>::zeros(),
            p: Matrix::<N, N>::zeros(),
        }
    }

    //https://www.mathworks.com/help/control/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html
    pub fn update<const M: usize>(
        &mut self,
        y: &Vector<M>,
        y_cov: &Matrix<M, M>,
        f: impl Fn(Vector<N>) -> Vector<M>,
    ) -> SKResult<()> {
        let c = self.alpha.powi(2) as f64 * (N as f64 + self.kappa);

        let cp = c.sqrt()
            * match self.p.cholesky() {
                Some(p) => p.l(),
                None => return skerror!("cannot take cholesky decomposition"),
            };

        let mut x_sigma_points = Vec::<Vector<N>>::with_capacity(2 * N + 1);

        // Create prior weights
        x_sigma_points.push(self.x.clone());
        for i in 0..N {
            x_sigma_points.push(self.x.clone() + cp.column(i));
        }
        for i in 0..N {
            x_sigma_points.push(self.x.clone() - cp.column(i));
        }

        // Compute predicted measurements
        let mut yhat_i = Vec::<Vector<M>>::with_capacity(2 * N + 1);
        for i in 0..2 * N + 1 {
            yhat_i.push(f(x_sigma_points[i]));
        }
        let yhat = yhat_i
            .iter()
            .enumerate()
            .fold(Vector::<M>::zeros(), |acc, (i, y)| {
                acc + self.weight_m[i] * y
            });

        // Compute predicted covariance
        let p_yy = yhat_i
            .iter()
            .enumerate()
            .fold(y_cov.clone(), |acc, (i, y)| {
                acc + self.weight_c[i] * (y - yhat) * (y - yhat).transpose()
            });

        // Compute cross covariance
        let p_xy = x_sigma_points
            .iter()
            .zip(yhat_i.iter())
            .enumerate()
            .fold(Matrix::<N, M>::zeros(), |acc, (i, (x, y))| {
                acc + self.weight_c[i] * (x - self.x) * (y - yhat).transpose()
            });

        let kalman_gain = p_xy * p_yy.try_inverse().unwrap();
        self.x = self.x + kalman_gain * (y - yhat);
        self.p = self.p - kalman_gain * p_yy * kalman_gain.transpose();
        Ok(())
    }

    /// Predict step
    /// Note: add your own process noise after this function is called
    pub fn predict(&mut self, f: impl Fn(Vector<N>) -> Vector<N>) {
        let c = self.alpha.powi(2) as f64 * (N as f64 + self.kappa);

        let cp = c.sqrt()
            * match self.p.cholesky() {
                Some(p) => p.l(),
                None => panic!("cannot take cholesky decomposition"),
            };
        let mut x_sigma_points = Vec::<Vector<N>>::with_capacity(2 * N + 1);

        // Create prior weights
        x_sigma_points.push(self.x.clone());
        for i in 0..N {
            x_sigma_points.push(self.x.clone() + cp.column(i));
        }
        for i in 0..N {
            x_sigma_points.push(self.x.clone() - cp.column(i));
        }

        // Compute predict with sigma values
        let mut x_post = Vec::<Vector<N>>::with_capacity(2 * N + 1);
        for i in 0..2 * N + 1 {
            x_post.push(f(x_sigma_points[i]));
        }

        // Update state
        self.x = x_post
            .iter()
            .enumerate()
            .fold(Vector::<N>::zeros(), |acc, (i, x)| {
                acc + self.weight_m[i] * x
            });

        // Update covariance
        self.p = x_post
            .iter()
            .enumerate()
            .fold(Matrix::<N, N>::zeros(), |acc, (i, x)| {
                acc + self.weight_c[i] * (x - self.x) * (x - self.x).transpose()
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand_distr::{Distribution, Normal};

    #[test]
    fn test_ukf() {
        let mut ukf = UKF::<2>::new_default();

        let normal = Normal::new(0.0, 1.0).unwrap();

        let ytruth = Vector::<2>::new(3.0, 4.0);
        let y_cov = Matrix::<2, 2>::new(1.0, 0.0, 0.0, 1.0);
        let v = normal.sample(&mut rand::thread_rng());
        let w = normal.sample(&mut rand::thread_rng());
        let ysample = ytruth + Vector::<2>::new(v, w);
        let offset = Vector::<2>::new(5.0, 8.0);
        let observe = |x: Vector<2>| x + offset;

        // Process noise
        let q = Matrix::<2, 2>::new(1.0e-12, 0.0, 0.0, 1.0e-12);
        ukf.x = ysample;
        ukf.p = y_cov;
        for _ix in 0..500 {
            let v = normal.sample(&mut rand::thread_rng());
            let w = normal.sample(&mut rand::thread_rng());
            let ysample = observe(ytruth + Vector::<2>::new(v, w));
            ukf.update(&ysample, &y_cov, observe).unwrap();
            ukf.p = ukf.p + q;
        }
    }
}
