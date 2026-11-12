# Satellite State

A satellite state at an epoch: position (m) and velocity (m/s) in the GCRF, with an optional 6×6 covariance and scheduled maneuvers, and a `propagate()` method that carries all of them forward with the [numerical propagator](satprop.md). Background: [State Vectors, STM & Covariance](../guide/satstate.md) and [Maneuver Coordinate Frames](../guide/maneuver_frames.md). Tutorials: [Covariance Propagation](../tutorials/Covariance%20Propagation.ipynb) and [Orbit Maneuvers](../tutorials/Orbit%20Maneuvers.ipynb).

::: satkit.satstate
