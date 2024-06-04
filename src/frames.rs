

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
            Frame::ITRF => write!(f, "ITRF"),
            Frame::TIRS => write!(f, "TIRS"),
            Frame::CIRS => write!(f, "CIRS"),
            Frame::GCRF => write!(f, "GCRF"),
            Frame::TEME => write!(f, "TEME"),
            Frame::EME2000 => write!(f, "EME2000"),
            Frame::ICRF => write!(f, "ICRF"),
            Frame::LVLH => write!(f, "LVLH"),
        }
    }
}

impl Frame {
    pub fn from_str(s: &str) -> Result<Self, &'static str> {
        match s {
            "ITRF" => Ok(Frame::ITRF),
            "TIRS" => Ok(Frame::TIRS),
            "CIRS" => Ok(Frame::CIRS),
            "GCRF" => Ok(Frame::GCRF),
            "TEME" => Ok(Frame::TEME),
            "EME2000" => Ok(Frame::EME2000),
            "ICRF" => Ok(Frame::ICRF),
            "LVLH" => Ok(Frame::LVLH),
            _ => Err("Invalid frame"),
        }
    }
}