# Moon

The `satkit.moon` module provides functions for computing the moon's position,
illumination fraction, and phase. The position is [Vallado (2013)](../guide/references.md#vallado2013) Algorithm 31 (§5.2.3, accurate to ~0.3° in ecliptic longitude); for higher precision use [`jplephem`](jplephem.md).

::: satkit.moon
