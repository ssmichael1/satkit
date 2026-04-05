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
    /// Radial / In-track / Cross-track.
    ///
    /// Also known as **RSW** (Vallado) or **RTN** (CCSDS) — the axes are
    /// identical; different communities use different names. Compile-time
    /// aliases [`Frame::RSW`] and [`Frame::RTN`] are available for
    /// readability when transcribing formulas from those conventions.
    ///
    /// * **R** (radial): unit vector along position, outward from Earth centre
    /// * **I** (in-track): perpendicular to R in the orbit plane, in the
    ///   prograde direction. **Not** parallel to the velocity vector unless
    ///   the flight-path angle is zero (circular orbit, or perigee/apogee of
    ///   an eccentric orbit). For eccentric orbits at non-apsidal true
    ///   anomaly, use [`Frame::NTW`] if you want an axis parallel to
    ///   velocity.
    /// * **C** (cross-track): along the angular momentum direction h = r × v,
    ///   completing the right-handed system
    ///
    /// This frame is the standard choice for **relative motion** (Hill /
    /// Clohessy-Wiltshire equations), for **CCSDS OEM / OMM covariance
    /// messages** (under the "RTN" name), and for **radial/normal burn
    /// components** whose physical meaning is tied to the position
    /// vector. `satkit`'s state-vector uncertainty API accepts RIC as
    /// one of several frames — see
    /// [`SatState::set_pos_uncertainty`](crate::orbitprop::SatState::set_pos_uncertainty).
    RIC,
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
    /// physically meaningful direction. Use [`Frame::RIC`] when you want
    /// "perpendicular to position" semantics or when interoperating with
    /// CCSDS OEM / OMM covariance messages (which use RIC under the
    /// "RTN" name).
    NTW,
}

impl Frame {
    /// Compile-time alias for [`Frame::RIC`] — Vallado's name for the same
    /// Radial / S=(W×R) / W=(R×V) orbital frame. Provided so code
    /// transcribing formulas from Vallado's *Fundamentals of Astrodynamics*
    /// can use the source's variable names verbatim. Semantically identical
    /// to `Frame::RIC`; `Display` always renders as `RIC`.
    pub const RSW: Self = Self::RIC;

    /// Compile-time alias for [`Frame::RIC`] — CCSDS OEM/ODM name for the
    /// same Radial / Tangential / Normal orbital frame. Semantically
    /// identical to `Frame::RIC`; `Display` always renders as `RIC`.
    pub const RTN: Self = Self::RIC;
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
            Self::RIC => write!(f, "RIC"),
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
            // RIC / RSW / RTN are three names for the same frame.
            "RIC" | "RSW" | "RTN" => Ok(Self::RIC),
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
    fn rsw_rtn_are_aliases_for_ric() {
        // Associated-constant aliases resolve to the same enum variant
        assert!(matches!(Frame::RSW, Frame::RIC));
        assert!(matches!(Frame::RTN, Frame::RIC));
        // Display is always the canonical name
        assert_eq!(Frame::RSW.to_string(), "RIC");
        assert_eq!(Frame::RTN.to_string(), "RIC");
    }

    #[test]
    fn from_str_accepts_aliases() {
        assert!(matches!(Frame::from_str("RSW").unwrap(), Frame::RIC));
        assert!(matches!(Frame::from_str("RTN").unwrap(), Frame::RIC));
        assert!(matches!(Frame::from_str("RIC").unwrap(), Frame::RIC));
        assert!(matches!(Frame::from_str("NTW").unwrap(), Frame::NTW));
    }
}
