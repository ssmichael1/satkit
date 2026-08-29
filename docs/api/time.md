# Time Representation

The `satkit` package makes use of a custom time class. This is a wrapper around a custom Rust class that adds the ability to represent time with different scales, or epochs, which is often necessary in the calculation of astronomical phenomena.

However, *all* functions in the `satkit` package that take time as an input can accept either the `satkit.time` class or the more commonly-used `datetime.datetime` class. For the latter, times are taken to have the *UTC* epoch.

The time scales and their relationships follow the IERS Conventions (2010), Chapter 10 ([Petit & Luzum 2010](../guide/references.md#petit2010)); UTC and leap seconds are defined by [ITU-R TF.460-6](../guide/references.md#itu460), with the leap-second table taken from [IERS Bulletin C](../guide/references.md#bulletinc).

::: satkit.timescale

::: satkit.time

::: satkit.duration
