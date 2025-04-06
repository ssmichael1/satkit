use anyhow::bail;

#[derive(Clone, Debug)]
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
    LVLH,
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
            _ => bail!("Invalid Frame"),
        }
    }
}
