"""CCSDS Orbit Ephemeris Message reader and writer (``satkit.OEM``)."""

from __future__ import annotations

import datetime
import math

import numpy as np
import pytest

import satkit as sk

A = 7.0e6
N = math.sqrt(sk.consts.mu_earth / A**3)
T0 = sk.time(2024, 1, 1)


def circular(t: float) -> np.ndarray:
    """State (m, m/s) on a circular orbit, ``t`` seconds after T0."""
    s, c = math.sin(N * t), math.cos(N * t)
    si, ci = math.sin(0.9), math.cos(0.9)
    return np.array([A * c, A * s * ci, A * s * si, -A * N * s, A * N * c * ci, A * N * c * si])


def kvn(n: int = 20, step: float = 60.0, meta: str = "", time_system: str = "UTC", acc: bool = False) -> str:
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        "COMMENT made by a test",
        "CREATION_DATE = 2024-01-01T00:00:00",
        "ORIGINATOR = TEST",
        "",
        "META_START",
        "COMMENT about the metadata",
        "OBJECT_NAME = CIRC",
        "OBJECT_ID = 2024-001A",
        "CENTER_NAME = EARTH",
        "REF_FRAME = EME2000",
        f"TIME_SYSTEM = {time_system}",
        "START_TIME = 2024-01-01T00:00:00",
        "STOP_TIME = 2024-01-02T00:00:00",
        *meta.splitlines(),
        "META_STOP",
        "",
        "COMMENT about the data",
    ]
    for i in range(n):
        t = i * step
        x = circular(t) / 1e3
        epoch = (datetime.datetime(2024, 1, 1) + datetime.timedelta(seconds=t)).isoformat()
        tail = " 0.001 0.002 0.003" if acc else ""
        lines.append(epoch + " " + " ".join(f"{v:.12f}" for v in x) + tail)
    return "\n".join(lines) + "\n"


COV = """COVARIANCE_START
EPOCH = 2024-01-01T00:00:00
COV_REF_FRAME = RTN
 1.0
 2.0  3.0
 4.0  5.0  6.0
 7.0  8.0  9.0 10.0
11.0 12.0 13.0 14.0 15.0
16.0 17.0 18.0 19.0 20.0 21.0
COVARIANCE_STOP
"""


def test_header_and_metadata():
    oem = sk.OEM.from_text(kvn(meta="INTERPOLATION = LAGRANGE\nINTERPOLATION_DEGREE = 5"))
    assert oem.version == "2.0"
    assert oem.originator == "TEST"
    assert oem.comments == ["made by a test"]
    assert oem.creation_date == T0
    assert oem.classification is None and oem.message_id is None
    assert len(oem) == 1 and len(oem.segments) == 1
    assert "CIRC" in repr(oem)

    seg = oem.segments[0]
    assert isinstance(seg, sk.OEMSegment)
    assert seg.object_name == "CIRC"
    assert seg.object_id == "2024-001A"
    assert seg.center_name == "EARTH"
    assert seg.ref_frame == "EME2000"
    assert seg.frame == sk.frame.EME2000
    assert seg.ref_frame_epoch is None
    assert seg.time_system == "UTC"
    assert seg.time_scale == sk.timescale.UTC
    assert seg.start_time == T0
    assert seg.stop_time == sk.time(2024, 1, 2)
    assert seg.useable_start_time is None and seg.useable_stop_time is None
    assert seg.interpolation == "LAGRANGE"
    assert seg.interpolation_degree == 5
    assert seg.metadata_comments == ["about the metadata"]
    assert seg.comments == ["about the data"]
    assert "EME2000" in repr(seg)


def test_states_are_si_arrays():
    seg = sk.OEM.from_text(kvn(n=5)).segments[0]
    assert len(seg) == 5
    assert seg.pos.shape == (5, 3) and seg.vel.shape == (5, 3)
    assert seg.pos.dtype == np.float64
    assert seg.acc is None
    assert len(seg.time) == 5 and all(isinstance(t, sk.time) for t in seg.time)
    assert seg.time[2] == T0 + sk.duration(seconds=120)
    for i in range(5):
        want = circular(60.0 * i)
        np.testing.assert_allclose(seg.pos[i], want[:3], atol=1e-6)
        np.testing.assert_allclose(seg.vel[i], want[3:], atol=1e-8)
    assert seg.usable_span == (T0, T0 + sk.duration(seconds=240))

    acc = sk.OEM.from_text(kvn(n=3, acc=True)).segments[0].acc
    np.testing.assert_allclose(acc, np.tile([1.0, 2.0, 3.0], (3, 1)))


def test_covariance():
    seg = sk.OEM.from_text(kvn(n=3) + COV).segments[0]
    (c,) = seg.covariances
    assert set(c) == {"time", "ref_frame", "cov", "comments"}
    assert c["time"] == T0
    assert c["ref_frame"] == "RTN"
    assert c["comments"] == []
    cov = c["cov"]
    assert cov.shape == (6, 6)
    np.testing.assert_array_equal(cov, cov.T)
    # km² → m²; lower triangle row by row
    assert cov[0, 0] == 1.0e6 and cov[1, 0] == 2.0e6 and cov[5, 5] == 21.0e6
    assert cov[5, 3] == 19.0e6 and cov[3, 5] == 19.0e6
    assert sk.OEM.from_text(kvn(n=3)).segments[0].covariances == []


@pytest.mark.parametrize(
    "meta",
    ["", "INTERPOLATION = LAGRANGE\nINTERPOLATION_DEGREE = 7", "INTERPOLATION = HERMITE\nINTERPOLATION_DEGREE = 7"],
)
def test_interp_accuracy(meta):
    oem = sk.OEM.from_text(kvn(n=40, meta=meta))
    for t in (0.0, 17.3, 30.0, 1111.1, 2330.5, 2340.0):
        state = oem.interp(T0 + sk.duration(seconds=t))
        assert state.shape == (6,)
        np.testing.assert_allclose(state[:3], circular(t)[:3], atol=1e-3)
        np.testing.assert_allclose(state[3:], circular(t)[3:], atol=1e-5)


def test_interp_arrays_and_time_types():
    oem = sk.OEM.from_text(kvn(n=40))
    seg = oem.segments[0]
    secs = [10.0, 500.0, 1500.5]
    times = [T0 + sk.duration(seconds=s) for s in secs]
    want = np.array([circular(s) for s in secs])

    out = oem.interp(times)
    assert out.shape == (3, 6)
    np.testing.assert_allclose(out[:, :3], want[:, :3], atol=1e-3)
    np.testing.assert_array_equal(out, seg.interp(times))
    np.testing.assert_array_equal(out, oem.interp(tuple(times)))
    np.testing.assert_array_equal(out[0], oem.interp(times[0]))
    assert oem.interp([]).shape == (0, 6)

    # datetime and datetime64 are accepted as times
    dt = datetime.datetime(2024, 1, 1, 0, 8, 20, tzinfo=datetime.timezone.utc)
    np.testing.assert_allclose(oem.interp(dt), out[1], atol=1e-6)
    np.testing.assert_allclose(oem.interp(np.datetime64("2024-01-01T00:08:20")), out[1], atol=1e-6)

    # A tabulated epoch is returned exactly
    np.testing.assert_array_equal(oem.interp(seg.time[7])[:3], seg.pos[7])


def test_interp_method_and_degree():
    oem = sk.OEM.from_text(kvn(n=10))
    t = T0 + sk.duration(seconds=270)
    want = circular(270.0)
    linear = oem.interp(t, method="lagrange", degree=1)
    sag = np.abs(linear[:3] - want[:3]).max()
    assert 100.0 < sag < 5.0e3
    cubic = oem.interp(t, method="Hermite", degree=3)
    assert np.abs(cubic[:3] - want[:3]).max() < 1.0
    # Only one of the two given: the other is the file's (Hermite, 7)
    np.testing.assert_array_equal(oem.interp(t, degree=7), oem.interp(t))
    np.testing.assert_array_equal(oem.interp(t, method="hermite"), oem.interp(t))
    np.testing.assert_array_equal(oem.interp(t, degree=1), oem.interp(t, method="hermite", degree=1))

    with pytest.raises(ValueError, match="lagrange"):
        oem.interp(t, method="spline")
    with pytest.raises(ValueError, match="at least 1"):
        oem.interp(t, degree=0)
    with pytest.raises(TypeError):
        oem.interp(t, "lagrange")  # keyword-only

    # A method satkit lacks is an error only when it would be used
    odd = sk.OEM.from_text(kvn(n=10, meta="INTERPOLATION = SPLINE"))
    with pytest.raises(ValueError, match="SPLINE"):
        odd.interp(t)
    np.testing.assert_array_equal(odd.interp(t, method="hermite"), oem.interp(t))


def test_interp_range_and_segments():
    first = kvn(n=10, meta="USEABLE_START_TIME = 2024-01-01T00:01:00\nUSEABLE_STOP_TIME = 2024-01-01T00:08:00")
    # Second segment: same orbit one hour later, appended without its header
    second = kvn(n=10).split("META_START", 1)[1].replace("2024-01-01T00:0", "2024-01-01T01:0")
    oem = sk.OEM.from_text(first + "META_START" + second)
    assert len(oem) == 2
    a, b = oem.segments
    assert a.usable_span == (T0 + sk.duration(minutes=1), T0 + sk.duration(minutes=8))
    assert b.time[0] == T0 + sk.duration(hours=1)

    with pytest.raises(ValueError, match="outside the usable span"):
        a.interp(T0 + sk.duration(seconds=30))
    with pytest.raises(ValueError, match="outside the usable span"):
        oem.interp(T0 + sk.duration(minutes=30))  # between the segments
    with pytest.raises(ValueError, match="outside the usable span"):
        oem.interp([T0 + sk.duration(minutes=2), T0 + sk.duration(days=2)])
    with pytest.raises(ValueError):
        a.interp(b.time[3])
    np.testing.assert_array_equal(oem.interp(b.time[3])[:3], b.pos[3])


def test_time_systems():
    utc = sk.OEM.from_text(kvn(n=2)).segments[0]
    gps = sk.OEM.from_text(kvn(n=2, time_system="GPS")).segments[0]
    assert gps.time_scale == sk.timescale.GPS
    # The same label in GPS time is 18 s earlier than in UTC (2024)
    assert (utc.time[0] - gps.time[0]).seconds == pytest.approx(18.0)
    assert gps.start_time == gps.time[0]
    with pytest.raises(ValueError, match="MET"):
        sk.OEM.from_text(kvn(n=2, time_system="MET"))


def test_xml():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<oem id="CCSDS_OEM_VERS" version="2.0">
  <header><CREATION_DATE>2024-01-01T00:00:00</CREATION_DATE><ORIGINATOR>TEST</ORIGINATOR></header>
  <body><segment>
    <metadata>
      <OBJECT_NAME>CIRC</OBJECT_NAME><OBJECT_ID>2024-001A</OBJECT_ID>
      <CENTER_NAME>EARTH</CENTER_NAME><REF_FRAME>ITRF2000</REF_FRAME><TIME_SYSTEM>UTC</TIME_SYSTEM>
      <START_TIME>2024-01-01T00:00:00</START_TIME><STOP_TIME>2024-01-01T00:01:00</STOP_TIME>
    </metadata>
    <data>
      <stateVector><EPOCH>2024-01-01T00:00:00</EPOCH>
        <X units="km">1</X><Y units="km">2</Y><Z units="km">3</Z>
        <X_DOT units="km/s">4</X_DOT><Y_DOT>5</Y_DOT><Z_DOT>6</Z_DOT></stateVector>
      <stateVector><EPOCH>2024-01-01T00:01:00</EPOCH>
        <X>7</X><Y>8</Y><Z>9</Z><X_DOT>10</X_DOT><Y_DOT>11</Y_DOT><Z_DOT>12</Z_DOT></stateVector>
    </data>
  </segment></body>
</oem>"""
    seg = sk.OEM.from_text(xml).segments[0]
    assert seg.frame == sk.frame.ITRF
    np.testing.assert_array_equal(seg.pos, [[1e3, 2e3, 3e3], [7e3, 8e3, 9e3]])
    np.testing.assert_array_equal(seg.vel, [[4e3, 5e3, 6e3], [10e3, 11e3, 12e3]])


def test_from_file(tmp_path):
    path = tmp_path / "circ.oem"
    path.write_text(kvn(n=4) + COV)
    for arg in (path, str(path)):
        oem = sk.OEM.from_file(arg)
        assert len(oem.segments[0]) == 4
        assert len(oem.segments[0].covariances) == 1
    missing = tmp_path / "nope.oem"
    with pytest.raises(FileNotFoundError) as e:
        sk.OEM.from_file(missing)
    assert e.value.filename == str(missing)


def test_parse_errors():
    with pytest.raises(ValueError, match="neither"):
        sk.OEM.from_text("1 25544U 98067A")
    with pytest.raises(ValueError, match=r"line 4.*ORIGINATR"):
        sk.OEM.from_text(kvn(n=2).replace("ORIGINATOR", "ORIGINATR"))
    with pytest.raises(ValueError, match="CENTER_NAME"):
        sk.OEM.from_text(kvn(n=2).replace("CENTER_NAME = EARTH\n", ""))
    with pytest.raises(ValueError, match="strictly increasing"):
        sk.OEM.from_text(kvn(n=2) + "2024-01-01T00:00:30 1 2 3 4 5 6\n")
    with pytest.raises(ValueError, match="found 5 fields"):
        sk.OEM.from_text(kvn(n=2) + "2024-01-01T01:00:00 1 2 3 4\n")
    with pytest.raises(ValueError, match="no state vectors"):
        sk.OEM.from_text(kvn(n=0))
    with pytest.raises(TypeError):
        sk.OEM.from_text(None)
    with pytest.raises(TypeError):
        sk.OEM()


def test_segments_outlive_the_message():
    seg = sk.OEM.from_text(kvn(n=3)).segments[0]
    assert seg.object_name == "CIRC" and seg.pos.shape == (3, 3)


def test_to_text_round_trip():
    oem = sk.OEM.from_text(kvn(n=5, meta="INTERPOLATION = LAGRANGE\nINTERPOLATION_DEGREE = 5", acc=True) + COV)
    text = oem.to_text()
    assert text.startswith("CCSDS_OEM_VERS = 2.0\nCOMMENT made by a test\n")
    assert "\n2024-01-01T00:01:00.000000 " in text
    back = sk.OEM.from_text(text)
    assert back.to_text() == text
    assert (back.comments, back.originator, back.creation_date) == (oem.comments, oem.originator, oem.creation_date)
    a, b = oem.segments[0], back.segments[0]
    for name in ("object_name", "object_id", "center_name", "ref_frame", "time_system", "start_time", "stop_time",
                 "interpolation", "interpolation_degree", "metadata_comments", "comments", "time"):
        assert getattr(a, name) == getattr(b, name), name
    np.testing.assert_allclose(b.pos, a.pos, rtol=0, atol=1e-6)
    np.testing.assert_allclose(b.vel, a.vel, rtol=0, atol=1e-9)
    np.testing.assert_allclose(b.acc, a.acc, rtol=0, atol=1e-11)
    assert len(b.covariances) == 1
    assert b.covariances[0]["ref_frame"] == "RTN"
    np.testing.assert_allclose(b.covariances[0]["cov"], a.covariances[0]["cov"], rtol=1e-13)


def test_to_text_time_system():
    text = sk.OEM.from_text(kvn(n=3, time_system="GPS")).to_text()
    assert "TIME_SYSTEM = GPS\nSTART_TIME = 2024-01-01T00:00:00.000000\n" in text


def test_to_file(tmp_path):
    oem = sk.OEM.from_text(kvn(n=4))
    path = tmp_path / "out.oem"
    assert oem.to_file(path) is None
    assert path.read_text() == oem.to_text()
    assert len(sk.OEM.from_file(str(path)).segments[0]) == 4
    with pytest.raises(OSError, match="nowhere"):
        oem.to_file(tmp_path / "nowhere" / "out.oem")


def _propagated(hours: float = 0.5, **settings):
    state = circular(0.0)
    return sk.propagate(
        state, T0, T0 + sk.duration(hours=hours), propsettings=sk.propsettings(gravity_degree=4, **settings)
    )


def test_from_propagation():
    res = _propagated()
    oem = sk.OEM.from_propagation(res, 60, object_name="SAT", object_id="2024-001A")
    assert oem.version == "2.0"
    assert oem.originator == "SATKIT"
    assert abs((oem.creation_date - sk.time.now()).seconds) < 60
    assert len(oem) == 1
    seg = oem.segments[0]
    assert (seg.object_name, seg.object_id, seg.center_name) == ("SAT", "2024-001A", "EARTH")
    assert seg.frame == sk.frame.GCRF
    assert seg.time_scale == sk.timescale.UTC
    assert (seg.interpolation, seg.interpolation_degree) == ("HERMITE", 7)
    assert (seg.start_time, seg.stop_time) == (res.time_begin, res.time_end)
    assert len(seg) == 31
    # The ends are the propagation's own states
    np.testing.assert_array_equal(np.concatenate([seg.pos[0], seg.vel[0]]), res.state_begin)
    np.testing.assert_array_equal(np.concatenate([seg.pos[-1], seg.vel[-1]]), res.state_end)

    # Written and read back, the table interpolates to the trajectory
    back = sk.OEM.from_text(oem.to_text())
    times = [T0 + sk.duration(seconds=s) for s in (17.0, 451.5, 1234.5, 1790.0)]
    want = np.array([res.interp(t) for t in times])
    got = back.interp(times)
    np.testing.assert_allclose(got[:, :3], want[:, :3], rtol=0, atol=1e-3)
    np.testing.assert_allclose(got[:, 3:], want[:, 3:], rtol=0, atol=1e-4)


def test_from_propagation_step_and_originator():
    res = _propagated()
    offsets = lambda oem: [(t - T0).seconds for t in oem.segments[0].time]  # noqa: E731
    kw = dict(object_name="S", object_id="I")
    # The end is there even when the step does not divide the span
    for step in (420, 420.0, sk.duration(minutes=7), datetime.timedelta(minutes=7)):
        assert offsets(sk.OEM.from_propagation(res, step, **kw)) == [0, 420, 840, 1260, 1680, 1800]
    assert offsets(sk.OEM.from_propagation(res, sk.duration(days=1), **kw)) == [0, 1800]
    assert sk.OEM.from_propagation(res, 60, originator="ME", **kw).originator == "ME"

    for bad in (0, -60, sk.duration(seconds=0)):
        with pytest.raises(ValueError, match="step must be positive"):
            sk.OEM.from_propagation(res, bad, **kw)
    for bad in ("60", None, True):
        with pytest.raises(TypeError, match="step must be"):
            sk.OEM.from_propagation(res, bad, **kw)
    with pytest.raises(TypeError):
        sk.OEM.from_propagation(res, 60, "S", "I")  # names are keyword-only
    with pytest.raises(TypeError):
        sk.OEM.from_propagation(res, 60)


def test_from_propagation_needs_dense_output():
    res = _propagated(enable_interp=False)
    assert not res.can_interp
    with pytest.raises(ValueError):
        sk.OEM.from_propagation(res, 60, object_name="S", object_id="I")


def test_from_propagation_with_stm():
    res = sk.propagate(
        circular(0.0), T0, T0 + sk.duration(minutes=30), output_phi=True,
        propsettings=sk.propsettings(gravity_degree=4),
    )
    seg = sk.OEM.from_propagation(res, 600, object_name="S", object_id="I").segments[0]
    assert len(seg) == 4
    np.testing.assert_array_equal(seg.pos[-1], res.state_end[:3])
