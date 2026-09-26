# Sun

The `satkit.sun` module provides functions for computing the sun's position
and related quantities such as sunrise/sunset times and satellite shadow status. The Sun position is the low-accuracy solar coordinates of [Meeus (1998)](../guide/references.md#meeus1998), Ch. 25, with the largest planetary and lunar terms of VSOP87 ([Bretagnon & Francou 1988](../guide/references.md#bretagnon1988)): within 3.6″ of JPL DE440 in longitude over 1900–2100. Sunrise/sunset is [Vallado (2013)](../guide/references.md#vallado2013) Algorithm 30 (§5.3.1), iterated at the event, optionally with the JPL ephemeris (`use_jpl=True`), and the shadow function is the conical umbra/penumbra model of [Montenbruck & Gill (2000)](../guide/references.md#montenbruck2000), §3.4.2.

::: satkit.sun
