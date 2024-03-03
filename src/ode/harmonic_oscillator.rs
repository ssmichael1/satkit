use super::types::{ODEResult, ODESystem};

pub struct HarmonicOscillator {
    k: f64,
}
impl HarmonicOscillator {
    pub fn new(k: f64) -> HarmonicOscillator {
        HarmonicOscillator { k: k }
    }
}

impl ODESystem for HarmonicOscillator {
    type Output = nalgebra::Vector2<f64>;
    fn ydot(&mut self, _x: f64, y: &Self::Output) -> ODEResult<Self::Output> {
        Ok(nalgebra::Vector2::<f64>::new(y[1], -self.k * y[0]))
    }
}
