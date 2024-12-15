use super::types::{ODEResult, ODESystem};

#[allow(unused)]
pub trait RKExplicit<const N: usize> {
    const A: [[f64; N]; N];
    const C: [f64; N];
    const B: [f64; N];

    fn step<S: ODESystem>(x0: f64, y0: &S::Output, h: f64, system: &mut S) -> ODEResult<S::Output> {
        let mut karr = Vec::<S::Output>::new();
        karr.push(system.ydot(x0, y0)?);

        // Create the "k"s
        for k in 1..N {
            karr.push(system.ydot(
                h.mul_add(Self::C[k], x0),
                &(karr.iter().enumerate().fold(y0.clone(), |acc, (idx, ki)| {
                    acc + ki.clone() * Self::A[k][idx] * h
                })),
            )?);
        }

        // Sum the "k"s
        Ok(karr
            .into_iter()
            .enumerate()
            .fold(y0.clone(), |acc, (idx, k)| acc + k * Self::B[idx] * h))
    }

    fn integrate<S: ODESystem>(
        x0: f64,
        xend: f64,
        dx: f64,
        y0: &S::Output,
        system: &mut S,
    ) -> ODEResult<Vec<S::Output>> {
        let mut x: f64 = x0;
        let mut v = Vec::new();
        let mut y = y0.clone();
        let steps = ((xend - x0) / dx).ceil() as usize;
        for _ in 0..steps {
            let ynew = Self::step(x, &y, dx, system)?;
            v.push(ynew.clone());
            x += dx;
            if x > xend {
                x = xend;
            }
            y = ynew;
        }
        Ok(v)
    }
}

pub struct RK4 {}
///
/// Buchter tableau for RK4
impl RKExplicit<4> for RK4 {
    const A: [[f64; 4]; 4] = [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    const B: [f64; 4] = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
    const C: [f64; 4] = [0.0, 0.5, 0.5, 1.0];
}

pub struct Midpoint {}
impl RKExplicit<2> for Midpoint {
    const A: [[f64; 2]; 2] = [[0.0, 0.0], [0.5, 0.0]];
    const B: [f64; 2] = [0.0, 1.0];
    const C: [f64; 2] = [0.0, 0.5];
}

#[cfg(test)]
mod tests {

    use super::*;
    type State = nalgebra::Vector2<f64>;

    struct HarmonicOscillator {
        k: f64,
    }
    impl HarmonicOscillator {
        const fn new(k: f64) -> Self {
            Self { k }
        }
    }

    impl ODESystem for HarmonicOscillator {
        type Output = nalgebra::Vector2<f64>;
        fn ydot(
            &mut self,
            _x: f64,
            y: &nalgebra::Vector2<f64>,
        ) -> ODEResult<nalgebra::Vector2<f64>> {
            Ok(State::new(y[1], -self.k * y[0]))
        }
    }

    #[test]
    fn testit() -> ODEResult<()> {
        let mut system = HarmonicOscillator::new(1.0);
        let y0 = State::new(1.0, 0.0);

        use std::f64::consts::PI;

        // integrating this harmonic oscillator between 0 and 2PI should return to the
        // original state
        let out2 = RK4::integrate(0.0, 2.0 * PI, 0.0001 * 2.0 * PI, &y0, &mut system)?;
        assert!((out2.last().unwrap()[0] - 1.0).abs() < 1.0e-6);
        assert!(out2.last().unwrap().abs()[1] < 1.0e-10);
        Ok(())
    }
}
