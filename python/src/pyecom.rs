use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pyo3::IntoPyObjectExt;

use satkit::orbitprop::EcomParams;

use anyhow::{bail, Result};

/// Python wrapper for ECOM solar-radiation-pressure coefficients
#[pyclass(name = "ecomparams", module = "satkit", from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct PyEcomParams(pub EcomParams);

const FIELDS: [&str; 13] = [
    "d0", "y0", "b0", "dc", "ds", "yc", "ys", "bc", "bs", "d2c", "d2s", "d4c", "d4s",
];

impl PyEcomParams {
    fn values(&self) -> [f64; 13] {
        let e = &self.0;
        [
            e.d0, e.y0, e.b0, e.dc, e.ds, e.yc, e.ys, e.bc, e.bs, e.d2c, e.d2s, e.d4c, e.d4s,
        ]
    }

    fn from_values(v: [f64; 13], sun_relative: bool) -> Self {
        Self(EcomParams {
            d0: v[0],
            y0: v[1],
            b0: v[2],
            dc: v[3],
            ds: v[4],
            yc: v[5],
            ys: v[6],
            bc: v[7],
            bs: v[8],
            d2c: v[9],
            d2s: v[10],
            d4c: v[11],
            d4s: v[12],
            sun_relative,
        })
    }
}

#[pymethods]
impl PyEcomParams {
    /// Create ECOM coefficients. All coefficients are in m/s^2 and default
    /// to zero; ``sun_relative`` selects the harmonic argument.
    ///
    /// Args:
    ///     d0, y0, b0 (float): constant D, Y, B terms
    ///     dc, ds (float): D cos φ, D sin φ
    ///     yc, ys (float): Y cos φ, Y sin φ
    ///     bc, bs (float): B cos φ, B sin φ
    ///     d2c, d2s, d4c, d4s (float): D cos 2φ, sin 2φ, cos 4φ, sin 4φ (ECOM2)
    ///     sun_relative (bool): False → φ = argument of latitude u (ECOM1);
    ///         True → φ = Δu measured from orbit noon (ECOM2)
    #[new]
    #[pyo3(signature = (*, d0=0.0, y0=0.0, b0=0.0, dc=0.0, ds=0.0, yc=0.0, ys=0.0, bc=0.0, bs=0.0, d2c=0.0, d2s=0.0, d4c=0.0, d4s=0.0, sun_relative=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        d0: f64,
        y0: f64,
        b0: f64,
        dc: f64,
        ds: f64,
        yc: f64,
        ys: f64,
        bc: f64,
        bs: f64,
        d2c: f64,
        d2s: f64,
        d4c: f64,
        d4s: f64,
        sun_relative: bool,
    ) -> Self {
        Self::from_values(
            [d0, y0, b0, dc, ds, yc, ys, bc, bs, d2c, d2s, d4c, d4s],
            sun_relative,
        )
    }

    /// Reduced ECOM1: D0, Y0, B0, Bc, Bs with harmonics in the argument of
    /// latitude (CODE's classic operational GPS set).
    #[staticmethod]
    fn reduced(d0: f64, y0: f64, b0: f64, bc: f64, bs: f64) -> Self {
        Self(EcomParams::reduced(d0, y0, b0, bc, bs))
    }

    /// Full 9-parameter ECOM1 (once-per-revolution terms on D, Y and B in
    /// the argument of latitude).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn ecom1(
        d0: f64,
        y0: f64,
        b0: f64,
        dc: f64,
        ds: f64,
        yc: f64,
        ys: f64,
        bc: f64,
        bs: f64,
    ) -> Self {
        Self(EcomParams::ecom1(d0, y0, b0, dc, ds, yc, ys, bc, bs))
    }

    /// ECOM2 (Arnold et al. 2015): D0, Y0, B0, B1c, B1s, D2c, D2s, D4c, D4s
    /// with harmonics in Δu from orbit noon.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn ecom2(
        d0: f64,
        y0: f64,
        b0: f64,
        b1c: f64,
        b1s: f64,
        d2c: f64,
        d2s: f64,
        d4c: f64,
        d4s: f64,
    ) -> Self {
        Self(EcomParams::ecom2(d0, y0, b0, b1c, b1s, d2c, d2s, d4c, d4s))
    }

    #[getter]
    fn d0(&self) -> f64 {
        self.0.d0
    }
    #[setter]
    fn set_d0(&mut self, v: f64) {
        self.0.d0 = v;
    }
    #[getter]
    fn y0(&self) -> f64 {
        self.0.y0
    }
    #[setter]
    fn set_y0(&mut self, v: f64) {
        self.0.y0 = v;
    }
    #[getter]
    fn b0(&self) -> f64 {
        self.0.b0
    }
    #[setter]
    fn set_b0(&mut self, v: f64) {
        self.0.b0 = v;
    }
    #[getter]
    fn dc(&self) -> f64 {
        self.0.dc
    }
    #[setter]
    fn set_dc(&mut self, v: f64) {
        self.0.dc = v;
    }
    #[getter]
    fn ds(&self) -> f64 {
        self.0.ds
    }
    #[setter]
    fn set_ds(&mut self, v: f64) {
        self.0.ds = v;
    }
    #[getter]
    fn yc(&self) -> f64 {
        self.0.yc
    }
    #[setter]
    fn set_yc(&mut self, v: f64) {
        self.0.yc = v;
    }
    #[getter]
    fn ys(&self) -> f64 {
        self.0.ys
    }
    #[setter]
    fn set_ys(&mut self, v: f64) {
        self.0.ys = v;
    }
    #[getter]
    fn bc(&self) -> f64 {
        self.0.bc
    }
    #[setter]
    fn set_bc(&mut self, v: f64) {
        self.0.bc = v;
    }
    #[getter]
    fn bs(&self) -> f64 {
        self.0.bs
    }
    #[setter]
    fn set_bs(&mut self, v: f64) {
        self.0.bs = v;
    }
    #[getter]
    fn d2c(&self) -> f64 {
        self.0.d2c
    }
    #[setter]
    fn set_d2c(&mut self, v: f64) {
        self.0.d2c = v;
    }
    #[getter]
    fn d2s(&self) -> f64 {
        self.0.d2s
    }
    #[setter]
    fn set_d2s(&mut self, v: f64) {
        self.0.d2s = v;
    }
    #[getter]
    fn d4c(&self) -> f64 {
        self.0.d4c
    }
    #[setter]
    fn set_d4c(&mut self, v: f64) {
        self.0.d4c = v;
    }
    #[getter]
    fn d4s(&self) -> f64 {
        self.0.d4s
    }
    #[setter]
    fn set_d4s(&mut self, v: f64) {
        self.0.d4s = v;
    }
    #[getter]
    fn sun_relative(&self) -> bool {
        self.0.sun_relative
    }
    #[setter]
    fn set_sun_relative(&mut self, v: bool) {
        self.0.sun_relative = v;
    }

    /// Coefficients as a dict (13 floats plus ``sun_relative``).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, v) in FIELDS.iter().zip(self.values()) {
            d.set_item(name, v)?;
        }
        d.set_item("sun_relative", self.0.sun_relative)?;
        Ok(d)
    }

    /// Build from a dict as produced by :meth:`to_dict`; missing keys default
    /// to zero / ``False``.
    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut v = [0.0f64; 13];
        for (slot, name) in v.iter_mut().zip(FIELDS) {
            if let Some(item) = d.get_item(name)? {
                *slot = item.extract()?;
            }
        }
        let sun_relative = match d.get_item("sun_relative")? {
            Some(item) => item.extract()?,
            None => false,
        };
        Ok(Self::from_values(v, sun_relative))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __repr__(&self) -> String {
        let mut parts: Vec<String> = FIELDS
            .iter()
            .zip(self.values())
            .filter(|(_, v)| *v != 0.0)
            .map(|(n, v)| format!("{n}={v:.4e}"))
            .collect();
        parts.push(format!(
            "sun_relative={}",
            if self.0.sun_relative { "True" } else { "False" }
        ));
        format!("ecomparams({})", parts.join(", "))
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Format v1: version byte, 13 little-endian f64, 1 byte sun_relative.
        let mut raw: Vec<u8> = Vec::with_capacity(1 + 13 * 8 + 1);
        raw.push(1u8);
        for v in self.values() {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        raw.push(u8::from(self.0.sun_relative));
        PyBytes::new(py, &raw).into_py_any(py)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyBytes>) -> Result<()> {
        let bytes = state.as_bytes(py);
        let (values, sun_relative) = decode_ecom_block(bytes)?;
        *self = Self::from_values(values, sun_relative);
        Ok(())
    }
}

/// Size of the serialized ECOM block (version byte + 13 f64 + flag).
pub const ECOM_BLOCK_LEN: usize = 1 + 13 * 8 + 1;

/// Serialize an `EcomParams` to the shared byte layout (used by both the
/// `ecomparams` pickle and the `satproperties` v2 pickle).
pub fn encode_ecom_block(e: &EcomParams) -> Vec<u8> {
    let mut raw = Vec::with_capacity(ECOM_BLOCK_LEN);
    raw.push(1u8);
    for v in PyEcomParams(*e).values() {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    raw.push(u8::from(e.sun_relative));
    raw
}

/// Inverse of [`encode_ecom_block`].
pub fn decode_ecom_block(bytes: &[u8]) -> Result<([f64; 13], bool)> {
    if bytes.len() != ECOM_BLOCK_LEN {
        bail!(
            "invalid ecomparams pickle: expected {} bytes, got {}",
            ECOM_BLOCK_LEN,
            bytes.len()
        );
    }
    if bytes[0] != 1 {
        bail!("unsupported ecomparams pickle version {}", bytes[0]);
    }
    let mut v = [0.0f64; 13];
    for (i, slot) in v.iter_mut().enumerate() {
        let at = 1 + i * 8;
        *slot = f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    }
    Ok((v, bytes[ECOM_BLOCK_LEN - 1] != 0))
}

pub fn ecom_from_block(bytes: &[u8]) -> Result<EcomParams> {
    let (v, s) = decode_ecom_block(bytes)?;
    Ok(PyEcomParams::from_values(v, s).0)
}
