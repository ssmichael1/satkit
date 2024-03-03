use super::rk_adaptive::RKAdaptive;

pub struct RKF45 {}
impl RKAdaptive<6> for RKF45 {
    const A: [[f64; 6]; 6] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0],
        [
            1932.0 / 2197.0,
            -7200.0 / 2197.0,
            7296.0 / 2197.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            439.0 / 216.0,
            -8.0,
            3680.0 / 513.0,
            -845.0 / 4104.0,
            0.0,
            0.0,
        ],
        [
            -8.0 / 27.0,
            2.0,
            -3544.0 / 2565.0,
            1859.0 / 4104.0,
            -11.0 / 40.0,
            0.0,
        ],
    ];
    const B: [f64; 6] = [
        16.0 / 135.0,
        0.0,
        6656.0 / 12825.0,
        28561.0 / 56430.0,
        -9.0 / 50.0,
        2.0 / 55.0,
    ];

    const BERR: [f64; 6] = {
        const BSTAR: [f64; 6] = [
            25.0 / 216.0,
            0.0,
            1408.0 / 2565.0,
            2197.0 / 4104.0,
            -0.2,
            0.0,
        ];
        let mut berr = [0.0; 6];
        let mut ix: usize = 0;
        while ix < 6 {
            berr[ix] = BSTAR[ix] - Self::B[ix];
            ix += 1;
        }
        berr
    };

    const C: [f64; 6] = [0.0, 0.25, 3.0 / 8.0, 12.0 / 13.0, 1.0, 0.5];

    const ORDER: usize = 4;

    const FSAL: bool = false;
}

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::super::RKAdaptiveSettings;
    use super::*;
    type State = nalgebra::Vector2<f64>;

    struct HarmonicOscillator {
        k: f64,
    }
    impl HarmonicOscillator {
        fn new(k: f64) -> HarmonicOscillator {
            HarmonicOscillator { k: k }
        }
    }

    impl ODESystem for HarmonicOscillator {
        type Output = nalgebra::Vector2<f64>;
        fn ydot(&mut self, _x: f64, y: &nalgebra::Vector2<f64>) -> ODEResult<Self::Output> {
            Ok(State::new(y[1], -self.k * y[0]))
        }
    }

    #[test]
    fn testit() {
        let mut system = HarmonicOscillator::new(1.0);
        let y0 = State::new(1.0, 0.0);

        use std::f64::consts::PI;

        let mut settings = RKAdaptiveSettings::default();
        settings.dense_output = false;

        // integrating this harmonic oscillator between 0 and 2PI should return to the
        // original state
        let res = RKF45::integrate(0.0, PI / 2.0, &y0, &mut system, &settings);
        println!("res = {:?}", res);
    }
}
