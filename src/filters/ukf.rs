//! Unscented Kalman Filter
//!
//! Uses nalgebra for state and covariance matrices


/// Generic float vector type of fixed size
type Vector<const T: usize> = nalgebra::SVector<f64, T>;

/// Generic float matrix type of fixed size
type Matrix<const M: usize, const N: usize> = nalgebra::SMatrix<f64, M, N>;
use anyhow::{anyhow, Result};

/// Unscented Kalman Filter
///
/// See: <https://www.mathworks.com/help/control/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html>
/// for a good explanation of the UKF algorithm
///
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
    /// Weights for the state estimate
    fn weight_m(alpha: f64, kappa: f64) -> Vec<f64> {
        let mut weight_m = Vec::<f64>::with_capacity(2 * N + 1);
        let den = alpha.powi(2) * (N as f64 + kappa);
        weight_m.push(1.0 - N as f64 / den);
        for _i in 1..2 * N + 1 {
            weight_m.push(1.0 / (2.0 * den));
        }
        weight_m
    }

    /// Weights for the covariance
    fn weight_c(alpha: f64, beta: f64, kappa: f64) -> Vec<f64> {
        let mut weight_c = Vec::<f64>::with_capacity(2 * N + 1);
        let den: f64 = alpha.powi(2) * (N as f64 + kappa);
        weight_c.push(2.0 + beta - alpha.powi(2).mul_add(1.0, N as f64 / den));
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

    /// <https://www.mathworks.com/help/control/ug/extended-and-unscented-kalman-filter-algorithms-for-online-state-estimation.html>
    /// Update state and covariance with new observation
    ///
    /// # Arguments
    /// * `y` - Observation vector
    /// * `y_cov` - Covariance of observation
    /// * `f` - Function to compute the observation from the state
    ///
    /// # Returns
    /// * `Result<()>` - Result of the update
    ///
    /// # Notes:
    /// * This will update the state estimate and the covariance matrix
    ///
    pub fn update<const M: usize>(
        &mut self,
        y: &Vector<M>,
        y_cov: &Matrix<M, M>,
        f: impl Fn(Vector<N>) -> Result<Vector<M>>,
    ) -> Result<()> {
        let c = self.alpha.powi(2) * (N as f64 + self.kappa);

        let cp = c.sqrt()
            * self
                .p
                .cholesky()
                .ok_or_else(|| anyhow!("Cannot take Cholesky decomposition"))?
                .l();

        let mut x_sigma_points = Vec::<Vector<N>>::with_capacity(2 * N + 1);

        // Create prior weights
        x_sigma_points.push(self.x);
        for i in 0..N {
            x_sigma_points.push(self.x + cp.column(i));
        }
        for i in 0..N {
            x_sigma_points.push(self.x - cp.column(i));
        }

        // Compute predict with sigma values
        let yhat_i = x_sigma_points
            .iter()
            .map(|x| f(*x))
            .collect::<Result<Vec<Vector<M>>>>()?;

        let yhat = yhat_i
            .iter()
            .enumerate()
            .fold(Vector::<M>::zeros(), |acc, (i, y)| {
                acc + self.weight_m[i] * y
            });

        // Compute predicted covariance
        let p_yy = yhat_i.iter().enumerate().fold(*y_cov, |acc, (i, y)| {
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

        let kalman_gain = p_xy
            * p_yy
                .try_inverse()
                .ok_or_else(|| anyhow!("Cannot take inverse of kalman gain; it is singular"))?;
        self.x += kalman_gain * (y - yhat);
        self.p -= kalman_gain * p_yy * kalman_gain.transpose();
        Ok(())
    }

    /// Predict step
    ///
    /// # Arguments
    /// * `f` - Function to compute the next state from the current state
    ///
    /// # Returns
    /// * Empty Ok value or an error
    ///
    /// # Notes
    /// * This will update the state estimate and the covariance matrix
    /// * This function does not add process noise to the state estimate,
    ///   you should add process noise after this function is called
    ///
    pub fn predict(&mut self, f: impl Fn(Vector<N>) -> Result<Vector<N>>) -> Result<()> {
        let c = self.alpha.powi(2) * (N as f64 + self.kappa);

        let cp = c.sqrt()
            * self
                .p
                .cholesky()
                .ok_or_else(|| anyhow!("Cannot take cholesky decomposition in predict step"))?
                .l();

        let mut x_sigma_points = Vec::<Vector<N>>::with_capacity(2 * N + 1);

        // Create prior weights
        x_sigma_points.push(self.x);
        for i in 0..N {
            x_sigma_points.push(self.x + cp.column(i));
        }
        for i in 0..N {
            x_sigma_points.push(self.x - cp.column(i));
        }

        // Compute predict with sigma values
        let x_post = x_sigma_points
            .iter()
            .map(|x| f(*x))
            .collect::<Result<Vec<Vector<N>>>>()?;

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

        Ok(())
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
        let v = normal.sample(&mut rand::rng());
        let w = normal.sample(&mut rand::rng());
        let ysample = ytruth + Vector::<2>::new(v, w);
        let offset = Vector::<2>::new(5.0, 8.0);
        let observe = |x: Vector<2>| Ok(x + offset);

        // Process noise
        let q = Matrix::<2, 2>::new(1.0e-12, 0.0, 0.0, 1.0e-12);
        ukf.x = ysample;
        ukf.p = y_cov;
        for _ix in 0..500 {
            let v = normal.sample(&mut rand::rng());
            let w = normal.sample(&mut rand::rng());
            let ysample = observe(ytruth + Vector::<2>::new(v, w)).unwrap();
            ukf.update(&ysample, &y_cov, observe).unwrap();
            ukf.p += q;
        }
    }
}
