use anyhow::bail;

/// Reference frame identifier.
///
/// Used throughout satkit to tag vectors with the coordinate system in which
/// they are expressed. For Earth-centred inertial and Earth-fixed frames the
/// definitions follow the IERS conventions; for spacecraft-local orbital
/// frames see the individual variant docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frame {
    /// International Terrestrial Reference Frame
    ITRF,
    /// Terrestrial Intermediate Reference System
    TIRS,
    /// Celestial Intermediate Reference System
    CIRS,
    /// Geocentric Celestial Reference Frame
    GCRF,
    /// True Equator Mean Equinox
    TEME,
    /// Earth Mean Equator 2000
    EME2000,
    /// International Celestial Reference Frame
    ICRF,
    /// Local Vertical Local Horizontal
    ///
    /// * z axis = -r (nadir)
    /// * y axis = -h (opposite angular momentum, h = r × v)
    /// * x axis completes the right-handed system (approximately velocity direction for circular orbits)
    LVLH,
    /// Radial / Tangential / Normal — CCSDS OEM/OMM/ODM convention.
    ///
    /// Also known as **RSW** (Vallado's *Fundamentals of Astrodynamics*)
    /// or **RIC** (Radial / In-track / Cross-track, older NASA usage and
    /// Clohessy-Wiltshire literature). The three names refer to the
    /// same axes; different communities use different letters. Compile-
    /// time aliases [`Frame::RSW`] and [`Frame::RIC`] are available so
    /// user code can spell the frame whichever way matches the source
    /// it's transcribing from — all three resolve to the same enum
    /// variant.
    ///
    /// * **R** (radial): unit vector along position, outward from Earth centre
    /// * **T** (tangential / in-track): perpendicular to R in the orbit
    ///   plane, in the prograde direction. **Not** parallel to the
    ///   velocity vector unless the flight-path angle is zero (circular
    ///   orbit, or perigee/apogee of an eccentric orbit). For eccentric
    ///   orbits at non-apsidal true anomaly, use [`Frame::NTW`] if you
    ///   want an axis parallel to velocity.
    /// * **N** (normal / cross-track): along the angular momentum
    ///   direction h = r × v, completing the right-handed system
    ///
    /// This frame is the standard choice for **CCSDS OEM / OMM / ODM
    /// orbit data messages** (which label it RTN), for **relative
    /// motion** formulations (Hill / Clohessy-Wiltshire equations, often
    /// written in RIC), and for **radial/normal burn components** whose
    /// physical meaning is tied to the position vector. `satkit`'s
    /// state-vector uncertainty API accepts RTN as one of several
    /// frames — see
    /// [`SatState::set_pos_uncertainty`](crate::orbitprop::SatState::set_pos_uncertainty).
    RTN,
    /// Velocity-aligned orbital frame (Vallado §3.3).
    ///
    /// * **N** (normal-to-velocity, in-plane): T̂ × Ŵ. For a circular orbit
    ///   this is the outward radial direction; for an eccentric orbit it
    ///   leans away from the radial by the flight-path angle.
    /// * **T** (tangent): v̂, the unit velocity vector
    /// * **W** (cross-track, out of plane): (r × v) / |r × v|, same as
    ///   RIC's C axis
    ///
    /// This is the natural frame for **thrust-along-velocity maneuvers**:
    /// a pure +T delta-v of magnitude Δv adds *exactly* Δv to |v| and
    /// changes orbital energy by v · Δv. For eccentric orbits at non-apsidal
    /// true anomalies, NTW and RIC differ by the flight-path angle, so the
    /// choice matters. Use NTW when planning prograde/retrograde burns,
    /// Hohmann transfers, or anything else where "along velocity" is the
    /// physically meaningful direction. Use [`Frame::RTN`] when you want
    /// "perpendicular to position" semantics or when interoperating with
    /// CCSDS OEM / OMM covariance messages (which use this frame under
    /// the RTN name).
    NTW,
}

impl Frame {
    /// Compile-time alias for [`Frame::RTN`] — Vallado's name for the same
    /// Radial / S=(W×R) / W=(R×V) orbital frame. Provided so code
    /// transcribing formulas from Vallado's *Fundamentals of Astrodynamics*
    /// can use the source's variable names verbatim. Semantically identical
    /// to `Frame::RTN`; `Display` always renders as `RTN`.
    pub const RSW: Self = Self::RTN;

    /// Compile-time alias for [`Frame::RTN`] — the older NASA /
    /// Clohessy-Wiltshire literature name for the same axes (Radial /
    /// In-track / Cross-track). Provided for backward compatibility with
    /// code that was written against the previous canonical name in
    /// earlier satkit versions, and for readers of papers that use
    /// "RIC" rather than "RTN". Semantically identical to `Frame::RTN`;
    /// `Display` always renders as `RTN`.
    pub const RIC: Self = Self::RTN;
}

impl std::fmt::Display for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::ITRF => write!(f, "ITRF"),
            Self::TIRS => write!(f, "TIRS"),
            Self::CIRS => write!(f, "CIRS"),
            Self::GCRF => write!(f, "GCRF"),
            Self::TEME => write!(f, "TEME"),
            Self::EME2000 => write!(f, "EME2000"),
            Self::ICRF => write!(f, "ICRF"),
            Self::LVLH => write!(f, "LVLH"),
            Self::RTN => write!(f, "RTN"),
            Self::NTW => write!(f, "NTW"),
        }
    }
}

impl std::str::FromStr for Frame {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ITRF" => Ok(Self::ITRF),
            "TIRS" => Ok(Self::TIRS),
            "CIRS" => Ok(Self::CIRS),
            "GCRF" => Ok(Self::GCRF),
            "TEME" => Ok(Self::TEME),
            "EME2000" => Ok(Self::EME2000),
            "ICRF" => Ok(Self::ICRF),
            "LVLH" => Ok(Self::LVLH),
            // RTN / RSW / RIC are three names for the same frame;
            // canonical is RTN.
            "RTN" | "RSW" | "RIC" => Ok(Self::RTN),
            "NTW" => Ok(Self::NTW),
            _ => bail!("Invalid Frame"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn ric_rsw_are_aliases_for_rtn() {
        // Associated-constant aliases resolve to the same enum variant
        assert!(matches!(Frame::RIC, Frame::RTN));
        assert!(matches!(Frame::RSW, Frame::RTN));
        // The three are equal as values (Frame derives PartialEq)
        assert_eq!(Frame::RIC, Frame::RTN);
        assert_eq!(Frame::RSW, Frame::RTN);
        // Display is always the canonical name (RTN)
        assert_eq!(Frame::RTN.to_string(), "RTN");
        assert_eq!(Frame::RIC.to_string(), "RTN");
        assert_eq!(Frame::RSW.to_string(), "RTN");
    }

    #[test]
    fn from_str_accepts_aliases() {
        assert!(matches!(Frame::from_str("RTN").unwrap(), Frame::RTN));
        assert!(matches!(Frame::from_str("RSW").unwrap(), Frame::RTN));
        assert!(matches!(Frame::from_str("RIC").unwrap(), Frame::RTN));
        assert!(matches!(Frame::from_str("NTW").unwrap(), Frame::NTW));
    }
}
