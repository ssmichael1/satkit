# Sun

The `satkit.sun` module provides functions for computing the sun's position
and related quantities such as sunrise/sunset times and satellite shadow status. The Sun position is [Vallado (2013)](../guide/references.md#vallado2013) Algorithm 29 (§5.1.1, accurate to ~0.01° over 1950–2050), sunrise/sunset is Algorithm 30 (§5.3.1), and the shadow function is the conical umbra/penumbra model of [Montenbruck & Gill (2000)](../guide/references.md#montenbruck2000), §3.4.2.

::: satkit.sun
