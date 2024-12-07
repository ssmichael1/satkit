use std::ops::{Add, Div, Mul, Sub};

use std::fmt::Debug;
use thiserror::Error;

use serde::{Deserialize, Serialize};

#[derive(Debug, Error)]
pub enum ODEError {
    #[error("Step error not finite")]
    StepErrorToSmall,
    #[error("No Dense Output in Solution")]
    NoDenseOutputInSolution,
    #[error("Interpolation exceeds solution bounds: {interp} not in [{start}, {stop}]")]
    InterpExceedsSolutionBounds { interp: f64, start: f64, stop: f64 },
    #[error("Interpolation not implemented for this integrator")]
    InterpNotImplemented,
    #[error("Y dot Function Error: {0}")]
    YDotError(String),
}

pub type ODEResult<T> = Result<T, ODEError>;

impl<T> From<ODEError> for ODEResult<T> {
    fn from(e: ODEError) -> Self {
        Err(e)
    }
}

pub trait ODEState:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + Clone
    + Sized
    + Debug
{
    // Element-wise divisior of self by other
    fn ode_elem_div(&self, other: &Self) -> Self;

    // Element-wise maximum of self with other
    fn ode_elem_max(&self, other: &Self) -> Self;

    // Euclideian norm
    fn ode_norm(&self) -> f64;

    // Element-wise absolute value
    fn ode_abs(&self) -> Self;

    // Add scalar to each element
    fn ode_scalar_add(&self, s: f64) -> Self;

    // Number of elements
    fn ode_nelem(&self) -> usize;

    // zero
    fn ode_zero() -> Self;
}

pub trait ODESystem {
    type Output: ODEState;
    fn ydot(&mut self, x: f64, y: &Self::Output) -> ODEResult<Self::Output>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseOutput<S>
where
    S: ODEState,
{
    pub x: Vec<f64>,
    pub h: Vec<f64>,
    pub yprime: Vec<Vec<S>>,
    pub y: Vec<S>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ODESolution<S>
where
    S: ODEState,
{
    pub nevals: usize,
    pub naccept: usize,
    pub nreject: usize,
    pub x: f64,
    pub y: S,
    pub dense: Option<DenseOutput<S>>,
}
