# Time Representation

The ``satkit`` package makes use of a custom time class.  This is wrapper around a custom ``rust`` class that adds the ability to reprsent time with different scales, or epochs, which is often necessary in the calcluation of astronomical phenomena.

However, *all* functions in the ``satkit`` package that take time as an input can accept either the ``satkit.time`` class or the more commonly-used ``datetime.datetime`` class.  For the latter, times are taken to have  the *UTC* epoch.

```{eval-rst}
.. autoapiclass:: satkit.timescale
   :members:

.. autoapiclass:: satkit.time
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. autoapiclass:: satkit.duration
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
```