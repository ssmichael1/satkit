# Moon

The `satkit.moon` module provides functions for computing the moon's position,
illumination fraction, and phase, moonrise and moonset times (`rise_set`), and the
times of the principal phases (`phase_times`, `next_phase`). The position is [Vallado (2013)](../guide/references.md#vallado2013) Algorithm 31 (§5.2.3, accurate to ~0.3° in ecliptic longitude); for higher precision use [`jplephem`](jplephem.md), or pass `use_jpl=True` to the rise/set and phase-time functions. The [Sun & Moon Rise/Set tutorial](../tutorials/riseset.ipynb) has worked examples.

::: satkit.moon
